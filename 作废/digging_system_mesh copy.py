import numpy as np
import mujoco
from scipy.spatial import cKDTree

class MeshDiggingSystem:
    def __init__(self, model, data, mesh_name, 
                 scene_body_name="voxel_target", 
                 voxel_xml_prefix="vx_", 
                 spacing=0.14, 
                 x_start=2.0,
                 manual_limit=None): # <--- 改动：默认为 None (自动计算)
        
        self.model = model
        self.data = data
        self.spacing = spacing
        self.box_size = spacing / 2
        self.wall_x_start = x_start
        self.scene_body_name = scene_body_name
        self.step_counter = 0
        self.last_head_pos = None
        
        # --- 0. 自动计算场景变换矩阵 ---
        self.scene_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene_body_name)
        if self.scene_body_id != -1:
            print(f"🌍 锁定场景锚点: '{scene_body_name}'")
            mujoco.mj_forward(model, data) 
            self.scene_pos = data.body(self.scene_body_id).xpos
            self.scene_rot = data.body(self.scene_body_id).xmat.reshape(3, 3)
            self.scene_rot_inv = self.scene_rot.T
        else:
            print(f"⚠️ 警告: 未找到 '{scene_body_name}'，假设位于原点。")
            self.scene_pos = np.array([0., 0., 0.])
            self.scene_rot_inv = np.eye(3)

        # --- 1. 获取 Mesh 顶点并建立 KD-Tree ---
        print(f"🔍 正在加载截割头 Mesh: '{mesh_name}'...")
        try:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            if mesh_id == -1:
                raise ValueError(f"❌ 找不到 Mesh: '{mesh_name}'")
            
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            self.mesh_verts = model.mesh_vert[vert_adr : vert_adr + vert_num * 3].reshape(-1, 3)
            
            # --- AABB 包围盒 ---
            self.aabb_min = np.min(self.mesh_verts, axis=0)
            self.aabb_max = np.max(self.mesh_verts, axis=0)
            dims = self.aabb_max - self.aabb_min
            
            # --- 🔥 关键修复：自动计算最大物理半径 ---
            # 计算 Mesh 上最远的一个点距离原点有多远
            self.real_mesh_radius = np.max(np.linalg.norm(self.mesh_verts, axis=1))
            
            # 设定判定阈值：实际半径 + 5cm 的容错空间
            # 这样既能保证尖端（实际半径处）能挖到，又能防止 3米外的噪点被挖到
            if manual_limit is not None:
                self.effective_limit = manual_limit
            else:
                self.effective_limit = self.real_mesh_radius + 0.05

            print(f"📏 Mesh 实际最大半径: {self.real_mesh_radius:.3f} 米")
            print(f"🛡️ 动态安全锁已设定为: {self.effective_limit:.3f} 米 (在此范围内的接触才有效)")

            # 异常检测
            if self.real_mesh_radius > 3.0:
                print("⚠️ 警告: Mesh 半径超过 3 米，请检查 STL 是否有飞离的噪点？")

            print("🌲 正在构建 KD-Tree...")
            self.kdtree = cKDTree(self.mesh_verts)
            print(f"✅ 初始化完成!")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

        # --- 2. 建立体素索引 ---
        self.voxel_index = {}
        self.active_voxels = set()
        print("🔍 正在索引体素...")
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        count = 0
        for i, name in enumerate(body_names):
            if name and name.startswith(voxel_xml_prefix):
                try:
                    parts = name.split('_')
                    k, i_idx, j = int(parts[1]), int(parts[2]), int(parts[3])
                    self.voxel_index[(k, i_idx, j)] = i
                    self.active_voxels.add((k, i_idx, j))
                    count += 1
                except: pass
        print(f"✅ 索引完成。体素数量: {count}")

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
        
        # 1. 粗筛 (Broad Phase)
        # 使用 effective_limit 作为搜索半径，确保尖端被包含
        search_range = int(np.ceil(self.effective_limit / self.spacing)) + 1
        
        center_k, center_i, center_j = self.world_to_local_grid(*head_pos)
        
        candidates = []
        candidate_keys = []
        
        for dk in range(-search_range, search_range + 1):
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    key = (center_k + dk, center_i + di, center_j + dj)
                    if key in self.active_voxels:
                        vid = self.voxel_index[key]
                        v_pos = self.data.body(vid).xpos
                        candidates.append(v_pos)
                        candidate_keys.append(key)
        
        if not candidates: return 0
        candidates = np.array(candidates)
        
        # 2. 坐标转换
        voxels_in_head_frame = (candidates - head_pos) @ head_mat 
        
        # --- 🔥 核心修复：基于实际尺寸的距离锁 ---
        dists_from_origin = np.linalg.norm(voxels_in_head_frame, axis=1)
        
        # 只过滤掉超出 (Mesh实际大小 + 5cm) 的点
        # 这样尖端（位于实际大小边缘）会被保留，而远处的噪点会被过滤
        is_within_limit = dists_from_origin <= self.effective_limit
        
        # --- AABB 校验 ---
        # 同时也放宽一点 AABB 的容差，确保尖端不被误剪
        aabb_tol = tolerance + 0.05 
        in_x = (voxels_in_head_frame[:, 0] >= self.aabb_min[0] - aabb_tol) & \
               (voxels_in_head_frame[:, 0] <= self.aabb_max[0] + aabb_tol)
        in_y = (voxels_in_head_frame[:, 1] >= self.aabb_min[1] - aabb_tol) & \
               (voxels_in_head_frame[:, 1] <= self.aabb_max[1] + aabb_tol)
        in_z = (voxels_in_head_frame[:, 2] >= self.aabb_min[2] - aabb_tol) & \
               (voxels_in_head_frame[:, 2] <= self.aabb_max[2] + aabb_tol)
        
        possible_mask = is_within_limit & in_x & in_y & in_z
        possible_indices = np.where(possible_mask)[0]
        
        final_hit = np.zeros(len(candidates), dtype=bool)
        
        # 3. 精筛 (KD-Tree)
        if len(possible_indices) > 0:
            dists, _ = self.kdtree.query(voxels_in_head_frame[possible_indices], k=1)
            hits = dists <= tolerance
            final_hit[possible_indices] = hits
            
        # 4. 删除
        indices_to_remove = np.where(final_hit)[0]
        for idx in indices_to_remove:
            key = candidate_keys[idx]
            if key in self.active_voxels:
                body_id = self.voxel_index[key]
                geom_id = self.model.body_geomadr[body_id]
                if geom_id != -1:
                    self.model.geom_size[geom_id] = [0, 0, 0]
                    self.model.geom_rgba[geom_id] = [0, 0, 0, 0]
                self.active_voxels.remove(key)
                reward += 1
                
        return reward

    def perform_cutting(self, cutting_body_name, tolerance=0.015): # 稍微调大一点容差以适应尖端
        self.step_counter += 1
        
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cutting_body_name)
        if body_id == -1: return 0
        
        current_pos = self.data.body(body_id).xpos.copy()
        current_mat = self.data.body(body_id).xmat.reshape(3, 3).copy()

        if self.last_head_pos is None:
            self.last_head_pos = current_pos
            return self._execute_single_cut(current_pos, current_mat, tolerance)

        dist = np.linalg.norm(current_pos - self.last_head_pos)
        
        TELEPORT_THRESHOLD = 0.15 
        step_size = self.spacing * 0.5 
        
        if dist > TELEPORT_THRESHOLD or dist < step_size:
            total_reward = self._execute_single_cut(current_pos, current_mat, tolerance)
        else:
            num_steps = int(np.ceil(dist / step_size))
            if num_steps > 15: num_steps = 15
            total_reward = 0
            for i in range(1, num_steps + 1):
                t = i / num_steps
                interp_pos = self.last_head_pos + (current_pos - self.last_head_pos) * t
                total_reward += self._execute_single_cut(interp_pos, current_mat, tolerance)

        self.last_head_pos = current_pos
        return total_reward