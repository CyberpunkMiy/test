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
                 # 参数: 椭球体半轴 [长轴(主轴)半径, 侧向半径1, 侧向半径2]
                 # 请务必把最大的半径写在第一个，代码会自动把它对齐到旋转轴上！
                 clean_threshold=[1.0, 0.5, 0.5]): 
        
        self.model = model
        self.data = data
        self.spacing = spacing
        self.box_size = spacing / 2
        self.wall_x_start = x_start
        self.scene_body_name = scene_body_name
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

        # --- 1. 加载 Mesh 并自动对齐旋转轴 ---
        print(f"🔍 正在加载截割头 Mesh: '{mesh_name}'...")
        try:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            if mesh_id == -1: raise ValueError(f"❌ 找不到 Mesh: '{mesh_name}'")
            
            # 获取原始顶点 (局部坐标)
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            raw_verts = model.mesh_vert[vert_adr : vert_adr + vert_num * 3].reshape(-1, 3)
            
            # ================= [核心修改: 自动对齐关节轴] =================
            radii = np.array(clean_threshold)
            radii[radii < 1e-6] = 1e-6

            # 1. 寻找关联的 Body 和 Joint
            # 通常 Mesh 名字和 Body 名字相似，或者通过 Geom 反查，这里我们假设外部传入的 mesh_name 对应的 body
            # 为了稳健，我们需要找到引用这个 Mesh 的 Body。
            # 这里简化逻辑：尝试用 Mesh 名字直接找 Body (这是常见命名习惯)
            # 如果找不到，就尝试找引用该 Mesh 的第一个 Geom 的 Body
            target_body_id = -1
            # 尝试直接按名字找 Body
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mesh_name)
            
            joint_axis = np.array([1.0, 0.0, 0.0]) # 默认 X 轴
            rotation_matrix = np.eye(3)
            found_joint = False

            if target_body_id != -1:
                # 查找该 Body 下的 Joint
                jnt_adr = model.body_jntadr[target_body_id]
                jnt_num = model.body_jntnum[target_body_id]
                
                if jnt_num > 0:
                    # 取第一个关节的轴向
                    joint_id = jnt_adr
                    # model.jnt_axis 存储的是局部坐标系下的轴向
                    joint_axis = model.jnt_axis[joint_id].copy()
                    print(f"⚙️ 检测到旋转关节轴 (局部): {joint_axis}")
                    found_joint = True
                else:
                    print("⚠️ 该 Body 没有关节，将默认使用 X 轴作为主轴。")
            else:
                 print(f"⚠️ 无法通过 Mesh 名 '{mesh_name}' 找到对应 Body，无法自动检测关节。")

            # 2. 计算旋转矩阵：将 [1, 0, 0] (椭球长轴) 对齐到 [joint_axis]
            # 只有当关节轴不是 X 轴时才需要计算
            target_axis = joint_axis / np.linalg.norm(joint_axis)
            source_axis = np.array([1.0, 0.0, 0.0])
            
            if found_joint and not np.allclose(target_axis, source_axis):
                # 计算两个向量之间的旋转
                # v_rot = rot * v_orig
                # 我们需要找到 rot 使得 rot * [1,0,0] = target_axis
                
                # 使用叉乘计算旋转轴
                rot_axis = np.cross(source_axis, target_axis)
                sin_theta = np.linalg.norm(rot_axis)
                cos_theta = np.dot(source_axis, target_axis)
                
                if sin_theta < 1e-6:
                    # 平行或反向
                    if cos_theta < 0: # 反向 180度
                        # 绕 Y 轴转 180
                        r = Rotation.from_euler('y', 180, degrees=True)
                        rotation_matrix = r.as_matrix()
                else:
                    rot_axis = rot_axis / sin_theta
                    theta = np.arctan2(sin_theta, cos_theta)
                    # 罗德里格斯公式生成旋转矩阵
                    r = Rotation.from_rotvec(rot_axis * theta)
                    rotation_matrix = r.as_matrix()
            
            # 3. 清洗过滤
            # 我们需要检查顶点 v 是否在椭球内。
            # 椭球定义在 X 轴上。我们的实际轴是 Joint Axis。
            # 所以我们要把 Mesh 顶点 "逆向旋转" 回 X 轴，再跟标准椭球比较。
            # v_aligned = R_inv @ v_raw  (因为 R 把 X 轴转到了 Joint 轴)
            aligned_verts = raw_verts @ rotation_matrix # 此时 rotation_matrix 是要把 X 转到 Joint
            # 等等，上面算出的 rotation_matrix 是  R_x_to_joint
            # 要把位于 Joint 轴的顶点转回 X 轴进行判定，应该乘 R.T (逆矩阵)
            # 但 raw_verts 是行向量 (N, 3)。 v_aligned = v @ R_inv.T = v @ R
            # 所以上面的 aligned_verts = raw_verts @ rotation_matrix.T 是对的吗？
            # 验证：v_joint_axis @ R.T = v_x_axis。
            # 行向量写法： (v_joint_axis * R^T) -> 错。
            # 正确推导： v_global = R * v_local. 
            # 我们希望 v_check = R_inv * v_raw.
            # 矩阵形式 (N,3): V_check = V_raw @ R_inv.T = V_raw @ R
            # 所以 aligned_verts = raw_verts @ rotation_matrix.T
            
            # 修正：将 Mesh 顶点逆旋转回标准 X 轴
            verts_to_check = raw_verts @ rotation_matrix 
            # 这里的数学关系比较绕，简而言之：我们求出了把 X 转到 关节轴 的 R。
            # 那么把 关节轴 转回 X，就是 R.T。
            # 行向量乘法： v @ R 相当于 列向量的 R.T @ v。
            # 所以如果 rotation_matrix 是 X->Joint。那我们需要 Joint->X。
            # 实际上 scipy 的 Rotation matrix 是坐标变换矩阵。
            
            # 让我们用一种更稳妥的方式：直接旋转 Debug Geom 即可，清洗计算用距离公式。
            # 计算点到直线的距离太麻烦，还是转回来简单。
            # 假设 rotation_matrix 把 [1,0,0] 变成了 joint_axis.
            # 那么 aligned_verts = raw_verts @ rotation_matrix 
            # 如果 raw_verts 在 joint_axis 上 (比如 [0,1,0]), R 是 X->Y (z轴90度).
            # [0,1,0] @ [[0,-1,0],[1,0,0],[0,0,1]] = [1, 0, 0]. 对了！
            # 所以直接乘是对的。
            
            # 再次确认：
            # 若 Mesh 在 Y 轴，Joint 也在 Y 轴。R (X->Y).
            # 我们想判定 Mesh 是否在 Y 轴范围内。
            # 方法：把 Mesh 旋转 -90 度到 X 轴，然后看 X 轴半径。
            # raw_verts ([0,1,0]) @ R_inv.T 
            # 还是直接用 scipy 简单：
            r_obj = Rotation.from_matrix(rotation_matrix)
            verts_to_check = r_obj.inv().apply(raw_verts) 

            # 4. 执行清洗
            if len(raw_verts) > 0:
                normalized = verts_to_check / radii
                dist_sq = np.sum(normalized**2, axis=1)
                
                valid_mask = dist_sq <= 1.0
                n_removed = len(raw_verts) - np.sum(valid_mask)
                
                if n_removed > 0:
                    print(f"🧹 [自动清洗] 移除了 {n_removed} 个异常顶点")
                    print(f"   >>> 判定形状: 椭球体 {radii} (已对齐关节轴)")
                    self.mesh_verts = raw_verts[valid_mask]
                else:
                    print(f"✨ Mesh 模型很干净 (已自动对齐关节轴检查)")
                    self.mesh_verts = raw_verts
            else:
                self.mesh_verts = raw_verts

            # ================= [可视化同步] =================
            debug_geom_name = "debug_clean_zone"
            debug_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, debug_geom_name)
            
            if debug_geom_id != -1:
                print(f"👀 同步可视化几何体 '{debug_geom_name}'...")
                model.geom_size[debug_geom_id] = radii
                # 几何体中心默认在原点
                model.geom_pos[debug_geom_id] = [0, 0, 0] 
                
                # 同步旋转
                # 刚才计算的 rotation_matrix 是把 X 轴转到 Joint 轴
                # 这正是 Geom 需要的姿态
                r = Rotation.from_matrix(rotation_matrix)
                q_xyzw = r.as_quat()
                # MuJoCo use w,x,y,z
                model.geom_quat[debug_geom_id] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
                print("   >>> 红色椭球体已对齐旋转轴")
            # ===============================================

            # 更新 KD-Tree
            if len(self.mesh_verts) > 0:
                self.max_radius = np.max(np.linalg.norm(self.mesh_verts, axis=1))
            else:
                self.max_radius = 0.1 

            print(f"📏 有效 Mesh 半径: {self.max_radius:.3f}m, 顶点数: {len(self.mesh_verts)}")
            self.kdtree = cKDTree(self.mesh_verts)
            print("✅ 初始化完成!")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise

        # --- 2. 体素索引 ---
        self.voxel_index = {}
        self.active_voxels = set()
        print("🔍 正在索引体素...")
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith(voxel_xml_prefix):
                try:
                    parts = name.split('_')
                    if len(parts) >= 4:
                        k, i_idx, j = int(parts[-3]), int(parts[-2]), int(parts[-1])
                        self.voxel_index[(k, i_idx, j)] = i
                        self.active_voxels.add((k, i_idx, j))
                except: pass
        print(f"✅ 索引完成。剩余体素: {len(self.active_voxels)}")

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
        center_k, center_i, center_j = self.world_to_local_grid(*head_pos)
        search_range = int(np.ceil(self.max_radius / self.spacing)) + 1
        
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
        
        voxels_in_head_frame = (candidates - head_pos) @ head_mat 
        dists, _ = self.kdtree.query(voxels_in_head_frame, k=1)
        
        is_hit = dists <= tolerance
        indices_to_remove = np.where(is_hit)[0]
        
        for idx in indices_to_remove:
            key = candidate_keys[idx]
            if key in self.active_voxels:
                body_id = self.voxel_index[key]
                geom_id = self.model.body_geomadr[body_id]
                if geom_id != -1:
                    self.model.geom_size[geom_id] = [0, 0, 0]
                    self.model.geom_rgba[geom_id] = [0, 0, 0, 0]
                    self.model.geom_conaffinity[geom_id] = 0
                    self.model.geom_contype[geom_id] = 0
                self.active_voxels.remove(key)
                reward += 1
        return reward

    def perform_cutting(self, cutting_body_name, tolerance=0.05):
        self.step_counter += 1
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
        if dist < step_size:
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
        self.last_head_mat = current_mat
        return total_reward
    
    def reset(self):
        """重置挖掘系统，恢复所有被消除的体素"""
        print("♻️ 重置体素墙...")
        # 1. 恢复 active_voxels 集合
        self.active_voxels = set(self.voxel_index.keys())
        
        # 2. 遍历所有体素 Body，恢复其 Geom 的属性
        for key, body_id in self.voxel_index.items():
            geom_id = self.model.body_geomadr[body_id]
            if geom_id != -1:
                # 恢复可见性 (假设原始 size 是 box_size, 需要根据你的 XML 确认)
                # 注意：这里假设是 Box，size对应长宽高的一半
                self.model.geom_size[geom_id] = [self.box_size, self.box_size, self.box_size] 
                
                # 恢复颜色 (这里设为默认颜色，例如红色或你的原始颜色)
                # 如果你想保留原始颜色，需要在 __init__ 里备份一下 model.geom_rgba
                self.model.geom_rgba[geom_id] = [0.8, 0.2, 0.2, 1.0] 
                
                # 恢复碰撞属性
                self.model.geom_conaffinity[geom_id] = 1
                self.model.geom_contype[geom_id] = 1
        
        # 重置计数器
        self.step_counter = 0
        self.last_head_pos = None
        
    def get_remaining_voxel_center(self):
        """获取剩余体素的质心（用于RL观察）"""
        if not self.active_voxels:
            return np.zeros(3)
        
        coords = []
        for key in self.active_voxels:
            body_id = self.voxel_index[key]
            coords.append(self.data.body(body_id).xpos)
        return np.mean(coords, axis=0)