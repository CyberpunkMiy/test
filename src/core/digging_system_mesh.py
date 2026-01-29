import numpy as np
import mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation 

class MeshDiggingSystem:
    def __init__(self, model, data, mesh_name, 
                 scene_body_name="imported_scene", 
                 voxel_xml_prefix="vx_", 
                 spacing=0.14, 
                 x_start=2.0,
                 clean_threshold=[1.0, 0.5, 0.5]): 
        
        self.model = model
        self.data = data
        self.spacing = spacing
        self.box_size = spacing / 2
        self.wall_x_start = x_start
        self.step_counter = 0
        self.last_head_pos = None
        self.last_head_mat = None

        # --- 0. 场景定位 ---
        self.scene_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene_body_name)
        if self.scene_body_id != -1:
            print(f"🌍 锁定场景锚点: '{scene_body_name}'")
            mujoco.mj_forward(model, data) 
            self.scene_pos = data.body(self.scene_body_id).xpos.copy()
            self.scene_rot = data.body(self.scene_body_id).xmat.reshape(3, 3).copy()
            self.scene_rot_inv = self.scene_rot.T
        else:
            self.scene_pos = np.array([0., 0., 0.])
            self.scene_rot_inv = np.eye(3)

        # --- 1. 加载截割头 Mesh 并处理清洗逻辑 ---
        # (保持原有的清洗逻辑不变)
        print(f"🔍 正在加载截割头 Mesh: '{mesh_name}'...")
        try:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            if mesh_id == -1: raise ValueError(f"❌ 找不到 Mesh: '{mesh_name}'")
            
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            raw_verts = model.mesh_vert[vert_adr : vert_adr + vert_num * 3].reshape(-1, 3)
            
            # --- 自动对齐与清洗 (沿用之前的逻辑) ---
            radii = np.array(clean_threshold)
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mesh_name)
            joint_axis = np.array([1.0, 0.0, 0.0]) 
            rotation_matrix = np.eye(3)

            if target_body_id != -1:
                jnt_adr = model.body_jntadr[target_body_id]
                if model.body_jntnum[target_body_id] > 0:
                    joint_axis = model.jnt_axis[jnt_adr].copy()

            target_axis = joint_axis / np.linalg.norm(joint_axis)
            source_axis = np.array([1.0, 0.0, 0.0])
            
            if not np.allclose(target_axis, source_axis):
                rot_axis = np.cross(source_axis, target_axis)
                sin_theta = np.linalg.norm(rot_axis)
                cos_theta = np.dot(source_axis, target_axis)
                if sin_theta < 1e-6:
                    if cos_theta < 0:
                        rotation_matrix = Rotation.from_euler('y', 180, degrees=True).as_matrix()
                else:
                    rot_axis = rot_axis / sin_theta
                    theta = np.arctan2(sin_theta, cos_theta)
                    rotation_matrix = Rotation.from_rotvec(rot_axis * theta).as_matrix()
            
            r_obj = Rotation.from_matrix(rotation_matrix)
            verts_to_check = r_obj.inv().apply(raw_verts) 
            
            if len(raw_verts) > 0:
                normalized = verts_to_check / radii
                dist_sq = np.sum(normalized**2, axis=1)
                valid_mask = dist_sq <= 1.0
                self.mesh_verts = raw_verts[valid_mask]
            else:
                self.mesh_verts = raw_verts

            # 更新 Mesh KDTree
            if len(self.mesh_verts) > 0:
                self.max_radius = np.max(np.linalg.norm(self.mesh_verts, axis=1))
                self.kdtree = cKDTree(self.mesh_verts)
            else:
                self.max_radius = 0.1
                self.kdtree = None
            
            print(f"✅ Mesh 初始化完成 (有效顶点: {len(self.mesh_verts)})")

        except Exception as e:
            print(f"❌ Mesh 初始化失败: {e}")
            raise

        # --- 2. 体素索引与【高速缓存】优化 ---
        self.voxel_index = {}      # Key -> Body ID
        self.active_voxels = set() # Set of Keys
        
        # 🆕 优化：构建静态位置缓存，避免每帧调用 MuJoCo API
        temp_centers = []
        self.voxel_key_to_idx = {} # Key -> Cache Array Index
        self.idx_to_body_id = []   # Cache Array Index -> Body ID
        
        print("🔍 正在索引体素并建立高速缓存...")
        idx_counter = 0
        
        # 必须先运行一次前向动力学以获取准确的世界坐标
        mujoco.mj_forward(model, data)
        
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith(voxel_xml_prefix):
                try:
                    parts = name.split('_')
                    if len(parts) >= 4:
                        k, i_idx, j = int(parts[-3]), int(parts[-2]), int(parts[-1])
                        key = (k, i_idx, j)
                        
                        self.voxel_index[key] = i
                        self.active_voxels.add(key)
                        
                        # 获取静态世界坐标并缓存
                        pos = data.body(i).xpos.copy()
                        temp_centers.append(pos)
                        
                        # 建立映射
                        self.voxel_key_to_idx[key] = idx_counter
                        self.idx_to_body_id.append(i)
                        idx_counter += 1
                except: pass
        
        # 转为 Numpy 数组进行向量化计算
        self.voxel_centers_cache = np.array(temp_centers, dtype=np.float32) # Shape: (N, 3)
        self.voxel_mask = np.ones(len(temp_centers), dtype=bool)            # Shape: (N,) True=Active
        
        print(f"✅ 索引完成。总计体素: {len(self.active_voxels)}")

    def world_to_local_grid(self, x, y, z):
        p_world = np.array([x, y, z])
        p_centered = p_world - self.scene_pos
        p_local = self.scene_rot_inv @ p_centered
        k = int(round((p_local[0] - self.wall_x_start - self.box_size) / self.spacing))
        i = int(round((p_local[2] - self.box_size) / self.spacing))
        j = int(round(p_local[1] / self.spacing))
        return k, i, j

    def _execute_single_cut(self, head_pos, head_mat, tolerance):
        reward = 0
        # 1. 粗筛：只检查附近的体素
        center_k, center_i, center_j = self.world_to_local_grid(*head_pos)
        search_range = int(np.ceil(self.max_radius / self.spacing)) + 1
        
        candidates_pos = []
        candidates_indices = [] # 记录在 cache 中的索引
        
        # 这里依然使用网格循环，因为这是 O(1) 的局部搜索，比全局 KDTree 更快
        for dk in range(-search_range, search_range + 1):
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    key = (center_k + dk, center_i + di, center_j + dj)
                    
                    # 检查是否还在 active 集合中
                    if key in self.active_voxels:
                        # 🆕 从缓存中取数据，不调 MuJoCo API
                        idx = self.voxel_key_to_idx[key]
                        pos = self.voxel_centers_cache[idx]
                        
                        candidates_pos.append(pos)
                        candidates_indices.append(idx)
        
        if not candidates_pos: return 0
        
        # 2. 精细判定：Mesh KDTree 查询
        candidates_pos = np.array(candidates_pos)
        # 将体素转换到截割头局部坐标系
        voxels_in_head_frame = (candidates_pos - head_pos) @ head_mat 
        
        if self.kdtree is None: return 0
        dists, _ = self.kdtree.query(voxels_in_head_frame, k=1)
        
        is_hit = dists <= tolerance
        hit_indices_in_candidates = np.where(is_hit)[0]
        
        # 3. 执行消除
        for hit_idx in hit_indices_in_candidates:
            cache_idx = candidates_indices[hit_idx] # 全局缓存索引
            
            # 双重检查掩码（防止重复计算）
            if self.voxel_mask[cache_idx]:
                # A. 视觉消除
                body_id = self.idx_to_body_id[cache_idx]
                geom_id = self.model.body_geomadr[body_id]
                if geom_id != -1:
                    self.model.geom_size[geom_id] = [0, 0, 0]
                    self.model.geom_rgba[geom_id] = [0, 0, 0, 0]
                    self.model.geom_conaffinity[geom_id] = 0
                    self.model.geom_contype[geom_id] = 0
                
                # B. 数据更新
                # 注意：active_voxels 集合还是要维护，因为 grid search 依赖它
                # 但主要计算依赖 voxel_mask
                
                # 通过 cache_idx 反查 key (比较耗时，但挖掘瞬间次数少，可接受)
                # 优化：也可以存 idx_to_key，但这里 candidates循环里其实可以传 key
                # 简单处理：因为我们 candidates 循环是基于 grid key 的，我们其实可以在 candidates 里存 key
                # 这里为了不改动太大，我们相信 cache_idx 唯一性
                
                self.voxel_mask[cache_idx] = False
                # 为了保持 grid 逻辑兼容，还是得从 set 里删掉
                # 这里稍微 tricky：我们需要 key。
                # 让我们在上一步 loop 里直接拿到 key
                pass 
        
        # 为了代码整洁，上面循环里拿到 idx 后，我们再遍历一遍 active_voxels 删除逻辑有点麻烦
        # 让我们回滚一点逻辑：在 candidates_pos 收集时顺便收集 keys
        pass
        
        # --- 重写循环部分 ---
        # 实际上 Python 的 list append 开销很小
        final_reward = 0
        candidate_keys = [] # 重新收集 key
        
        # 重新运行上面的 Loop (为了清晰逻辑，合并写在一起更好，但为了 patch 简单)：
        # 其实在上面那个 Loop 里：
        # candidates_indices.append(idx) 后面加一句 candidate_keys.append(key) 即可
        # 假设我们加了... (下面是修正后的完整逻辑)
        
        pass # (占位符)

        return reward # (占位符)
    
    # --- 修正后的 perform_cutting 逻辑 (覆盖上面的 _execute_single_cut) ---
    def _execute_single_cut(self, head_pos, head_mat, tolerance):
        reward = 0
        center_k, center_i, center_j = self.world_to_local_grid(*head_pos)
        search_range = int(np.ceil(self.max_radius / self.spacing)) + 1
        
        candidates_pos = []
        candidates_keys = []
        candidates_indices = []
        
        for dk in range(-search_range, search_range + 1):
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    key = (center_k + dk, center_i + di, center_j + dj)
                    if key in self.active_voxels:
                        idx = self.voxel_key_to_idx[key]
                        # 🆕 极速读取
                        candidates_pos.append(self.voxel_centers_cache[idx])
                        candidates_keys.append(key)
                        candidates_indices.append(idx)
        
        if not candidates_pos: return 0
        
        candidates_pos = np.array(candidates_pos)
        voxels_in_head_frame = (candidates_pos - head_pos) @ head_mat 
        
        if self.kdtree is None: return 0
        dists, _ = self.kdtree.query(voxels_in_head_frame, k=1)
        
        hit_mask = dists <= tolerance
        hit_indices = np.where(hit_mask)[0]
        
        for i in hit_indices:
            key = candidates_keys[i]
            cache_idx = candidates_indices[i]
            
            # 双重确认
            if key in self.active_voxels:
                # 1. 视觉消除
                body_id = self.idx_to_body_id[cache_idx]
                geom_id = self.model.body_geomadr[body_id]
                if geom_id != -1:
                    self.model.geom_size[geom_id] = [0, 0, 0]
                    self.model.geom_rgba[geom_id] = [0, 0, 0, 0]
                    self.model.geom_conaffinity[geom_id] = 0
                    self.model.geom_contype[geom_id] = 0
                
                # 2. 逻辑消除
                self.active_voxels.remove(key)
                self.voxel_mask[cache_idx] = False # 🆕 更新掩码
                reward += 1
                
        return reward

    def perform_cutting(self, cutting_body_name, tolerance=0.05):
        self.step_counter += 1
        # 获取截割头位置 (这部分很快，不需要优化)
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cutting_body_name)
            if body_id != -1:
                current_pos = self.data.body(body_id).xpos.copy()
                current_mat = self.data.body(body_id).xmat.reshape(3, 3).copy()
            else:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cutting_body_name)
                if site_id == -1: return 0
                current_pos = self.data.site_xpos[site_id].copy()
                current_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        except: return 0

        if self.last_head_pos is None:
            self.last_head_pos = current_pos
            self.last_head_mat = current_mat
            return self._execute_single_cut(current_pos, current_mat, tolerance)

        dist = np.linalg.norm(current_pos - self.last_head_pos)
        step_size = self.spacing * 0.5 
        
        total_reward = 0
        if dist < step_size:
            total_reward = self._execute_single_cut(current_pos, current_mat, tolerance)
        else:
            num_steps = int(np.ceil(dist / step_size))
            if num_steps > 10: num_steps = 10 # 限制步数防止卡顿
            for i in range(1, num_steps + 1):
                t = i / num_steps
                interp_pos = self.last_head_pos + (current_pos - self.last_head_pos) * t
                total_reward += self._execute_single_cut(interp_pos, current_mat, tolerance)

        self.last_head_pos = current_pos
        self.last_head_mat = current_mat
        return total_reward
    
    def get_local_target(self, head_pos, k=50):
        """
        【RL 核心优化】获取局部密度中心
        极速版：使用 Numpy 掩码和缓存，避免循环 API 调用
        """
        # 如果没有剩余体素
        if not np.any(self.voxel_mask):
            return np.zeros(3)

        # 🆕 1. 直接从缓存中获取所有 Active 的坐标 (极快)
        # self.voxel_mask 维护了当前存活的体素
        active_points = self.voxel_centers_cache[self.voxel_mask]
        
        if len(active_points) == 0:
            return np.zeros(3)
        
        # 2. 如果剩余数量少于 k，直接返回中心
        if len(active_points) <= k:
            return np.mean(active_points, axis=0)
            
        # 3. 向量化计算距离
        diff = active_points - head_pos
        dist_sq = np.sum(diff**2, axis=1)
        
        # 4. 找到最近的 k 个 (O(N) 复杂度)
        nearest_indices = np.argpartition(dist_sq, k)[:k]
        
        # 5. 计算中心
        local_target = np.mean(active_points[nearest_indices], axis=0)
        
        return local_target

    def get_remaining_voxel_center(self):
        """兼容性接口：获取全局质心"""
        if not np.any(self.voxel_mask):
            return np.zeros(3)
        return np.mean(self.voxel_centers_cache[self.voxel_mask], axis=0)

    def reset(self):
        """重置挖掘系统"""
        print("♻️ 重置体素墙...")
        
        # 1. 恢复集合
        self.active_voxels = set(self.voxel_index.keys())
        
        # 2. 恢复掩码 (全部设为 True)
        self.voxel_mask[:] = True
        
        # 3. 视觉恢复
        # 这里只能循环了，因为修改 model 属性没有批量接口
        for i, body_id in enumerate(self.idx_to_body_id):
            geom_id = self.model.body_geomadr[body_id]
            if geom_id != -1:
                self.model.geom_size[geom_id] = [self.box_size, self.box_size, self.box_size] 
                self.model.geom_rgba[geom_id] = [0.8, 0.2, 0.2, 1.0] 
                self.model.geom_conaffinity[geom_id] = 1
                self.model.geom_contype[geom_id] = 1
        
        self.step_counter = 0
        self.last_head_pos = None