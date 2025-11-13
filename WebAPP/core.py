import ase
import ase.io
import ase.build
import numpy
import logging
import os
import json
import scipy.spatial
import argparse # Import argparse
from chgnet.model import CHGNetCalculator
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation 
from itertools import combinations
from scipy.spatial import cKDTree
import alphashape
from scipy.spatial import ConvexHull
 
# --- 日志配置 (This will now be controlled by the handlers) ---
# We will set up the handlers in the main block to allow file logging
logger = logging.getLogger("SurfaceWorkflow")
logger.setLevel(logging.INFO)
# Prevent duplicate logs if already configured
if not logger.handlers:
    logger.addHandler(logging.StreamHandler()) # Log to console

class SurfaceAdsorptionWorkflow:
    def __init__(self, **kwargs):
        # All parameters are now passed in, making the class self-contained
        self.output_folder = kwargs.get("output_folder", "adsorption_results")
        self.logger = logger # Use the globally configured logger
        self.surface_axis = int(kwargs.get("surface_axis", 2))
        self.place_on_bottom = bool(kwargs.get("place_on_bottom", False))
        self.adsorption_height = float(kwargs.get("adsorption_height", 2.0))
        self.vacuum_thickness = float(kwargs.get("vacuum_thickness", 20.0))
        self.surface_search_depth = float(kwargs.get("surface_search_depth", 3.5))
        self.collision_threshold = float(kwargs.get("collision_threshold", 1.2))
        self.hollow_sites_enabled = bool(kwargs.get("hollow_sites_enabled", True))
        self.knn_neighbors = int(kwargs.get("knn_neighbors", 2))
        self.hollow_site_deduplication_distance = float(kwargs.get("hollow_site_deduplication_distance", 1.5))
        self.on_top_sites_enabled = bool(kwargs.get("on_top_sites_enabled", True))
        self.on_top_target_atom = str(kwargs.get("on_top_target_atom", 'O'))
        
        self.rotation_count = int(kwargs.get("rotation_count", 50))
        self.rotation_step = float(kwargs.get("rotation_step", 30))
        
        self.rotation_method = bool(kwargs.get("rotation_method", False))


        
        os.makedirs(self.output_folder, exist_ok=True)
        self.logger.info(f"Workflow initialized with parameters: {kwargs}")

    def run(self, substrate_path: str, adsorbate_path: str):
        try:
            substrate, adsorbate = self._load_structures(substrate_path, adsorbate_path)
            surface_slab = self._build_surface_slab(substrate)
            
            calculator = CHGNetCalculator()
            slab_for_calc = surface_slab.copy()
            slab_for_calc.calc = calculator
            e_slab = slab_for_calc.get_potential_energy()
            self.logger.info(f"基底能量: {e_slab:.4f} eV")
            adsorbate_for_calc = adsorbate.copy()
            adsorbate_for_calc.calc = calculator
        
            e_adsorbate = adsorbate_for_calc.get_potential_energy()
            self.logger.info(f"吸附物能量: {e_adsorbate:.4f} eV")

            surface_atoms_coords, surface_atoms_indices = self._find_surface_atoms(surface_slab)
            if surface_atoms_coords.size == 0: 
                self.logger.warning("未找到表面原子，退出计算。")
                return

            unified_sites_data = self._find_adsorption_sites(surface_slab, surface_atoms_coords, surface_atoms_indices)
            if not unified_sites_data: 
                self.logger.warning("未找到有效的吸附位点，退出计算。")
                return
            
            optimized_results, total_sites = self._place_and_optimize_adsorbate(
                surface_slab, adsorbate, unified_sites_data, e_slab, e_adsorbate
            )
            
            if optimized_results:
                self._create_json_for_frontend_visualization(surface_slab, optimized_results)
                self._visualize_final_results(surface_slab, optimized_results)
            else:
                self.logger.warning("没有任何位点成功优化，无法生成最终结果文件。")
        except Exception as e:
            self.logger.error(f"工作流发生严重错误: {e}", exc_info=True)
            raise
        self.logger.info("Calculation finished successfully.")

    def _find_surface_atoms(self, slab: ase.Atoms) -> tuple[numpy.ndarray, numpy.ndarray]:
        placement = "底层(坐标最小侧)" if self.place_on_bottom else "顶层(坐标最大侧)"
        self.logger.info(f"寻找{placement}原子...")
        positions = slab.get_positions()
        coords_on_axis = positions[:, self.surface_axis]
        
        if self.place_on_bottom:
            surface_level = numpy.min(coords_on_axis)
            mask = (coords_on_axis <= surface_level + self.surface_search_depth)
        else:
            surface_level = numpy.max(coords_on_axis)
            mask = (coords_on_axis >= surface_level - self.surface_search_depth)
            
        surface_atoms_indices = numpy.where(mask)[0]
        surface_atoms_coords = positions[surface_atoms_indices]
        self.logger.info(f"找到 {len(surface_atoms_coords)} 个{placement}原子。")
        
        # --- NEW LINE TO SAVE SURFACE ATOMS FOR VISUALIZATION ---
        with open(os.path.join(self.output_folder, "surface_atoms.json"), 'w') as f:
            json.dump({'coords': surface_atoms_coords.tolist()}, f)
        # --- END OF NEW LINE ---

        return surface_atoms_coords, surface_atoms_indices
        
    # All other methods (_load_structures, _build_surface_slab, etc.) remain IDENTICAL to your original version.
    # ...
    # PASTE ALL YOUR OTHER METHODS HERE, UNCHANGED
    # ...
    # (The following are placeholders for brevity, use your full original methods)
    def _load_structures(self, substrate_path: str, adsorbate_path: str) -> tuple[ase.Atoms, ase.Atoms]:
        self.logger.info("加载结构文件...")
        substrate = ase.io.read(substrate_path,format="cif")
        adsorbate = ase.io.read(adsorbate_path,format="cif")
        self.logger.info(f"加载成功: 基板={substrate.get_chemical_formula()} | 吸附物={adsorbate.get_chemical_formula()}")
        return substrate, adsorbate

    def _build_surface_slab(self, substrate: ase.Atoms) -> ase.Atoms:
        self.logger.info("构建表面平板模型...")
        slab = substrate.copy()
        slab.set_pbc(True)
        slab.center(vacuum=self.vacuum_thickness, axis=self.surface_axis)
        pbc = [True, True, True]
        pbc[self.surface_axis] = False
        slab.set_pbc(pbc)
        self.logger.info(f"平板构建完成。PBC: {slab.get_pbc()} | 晶胞尺寸: {numpy.diag(slab.cell)}")
        ase.io.write(os.path.join(self.output_folder, "01_built_surface.cif"), slab)
        return slab

    def _find_adsorption_sites(self, slab: ase.Atoms, surface_atoms_coords: numpy.ndarray, surface_atoms_indices: numpy.ndarray) -> list:
        hollow_sites_data = self._find_hollow_sites(surface_atoms_coords) if self.hollow_sites_enabled else []
        on_top_sites_data = self._find_on_top_sites(slab, surface_atoms_indices) if self.on_top_sites_enabled else []
        self.logger.info(f"空穴hollow点有{hollow_sites_data}")
        self.logger.info(f"顶位on_top点有{on_top_sites_data}")
        all_sites_data = hollow_sites_data + on_top_sites_data
        self.logger.info(f"位点搜索完成：{len(hollow_sites_data)} 个空腔位, {len(on_top_sites_data)} 个顶位, 总计 {len(all_sites_data)} 个。")
        return all_sites_data

    def _find_hollow_sites(self, surface_atoms_coords: numpy.ndarray) -> list:
        self.logger.info("-> 计算“空腔”位点...")
        if len(surface_atoms_coords) < self.knn_neighbors + 1: return []
        plane_axes = [i for i in range(3) if i != self.surface_axis]
        coords_2d = surface_atoms_coords[:, plane_axes]
        kdtree = scipy.spatial.KDTree(coords_2d)
        potential_sites_data, away_direction_vec = [], numpy.zeros(3)
        away_direction_vec[self.surface_axis] = -1.0 if self.place_on_bottom else 1.0
        _, indices_list = kdtree.query(coords_2d, k=self.knn_neighbors + 1)
        for i in range(len(surface_atoms_coords)):
            cluster_atoms_3d = surface_atoms_coords[indices_list[i]]
            centroid_3d = numpy.mean(cluster_atoms_3d, axis=0)
            normal = away_direction_vec.copy()
            if self.knn_neighbors >= 2 and len(cluster_atoms_3d) > 2:
                v1, v2, v3 = cluster_atoms_3d[0], cluster_atoms_3d[1], cluster_atoms_3d[2]
                cross_product = numpy.cross(v2 - v1, v3 - v1)
                norm_val = numpy.linalg.norm(cross_product)
                if norm_val > 1e-6:
                    cross_product /= norm_val
                    if numpy.dot(cross_product, away_direction_vec) < 0:
                        cross_product = -cross_product
                    normal = cross_product
            potential_sites_data.append({'site': centroid_3d, 'normal': normal, 'type': 'Hollow'})
        unique_sites_data = []
        if potential_sites_data:
            unique_sites_data.append(potential_sites_data[0])
            for site_data in potential_sites_data[1:]:
                if not any(numpy.linalg.norm(site_data['site'] - usd['site']) < self.hollow_site_deduplication_distance for usd in unique_sites_data):
                    unique_sites_data.append(site_data)
        self.logger.info(f"--> 去重后得到 {len(unique_sites_data)} 个空腔位点。")
        return unique_sites_data
    
    # def find_surface_hollow_sites_3d(self, surface_coords, neighbor_cut=4.0):
    #     tree = cKDTree(surface_coords)
    #     hollow_sites = []
    #     for idx, p in enumerate(surface_coords):
    #         neighbors = tree.query_ball_point(p, neighbor_cut)
    #         neighbors = [n for n in neighbors if n != idx]
    #         for c in combinations(neighbors, 2):
    #             i, j = idx, c[0]
    #             k = c[1]
    #             v1, v2, v3 = surface_coords[i], surface_coords[j], surface_coords[k]
    #             cp = numpy.cross(v2 - v1, v3 - v1)
    #             norm_val = numpy.linalg.norm(cp)
    #             if norm_val < 1e-6:
    #                 # fallback: 默认法向量，确保是单位向量
    #                 normal = numpy.zeros(3)
    #                 normal[self.surface_axis] = -1.0 if self.place_on_bottom else 1.0
    #             else:
    #                 normal = cp / norm_val
    #                 # 朝向修正
    #                 away_direction_vec = numpy.zeros(3)
    #                 away_direction_vec[self.surface_axis] = -1.0 if self.place_on_bottom else 1.0
    #                 if numpy.dot(normal, away_direction_vec) < 0:
    #                     normal = -normal
    #             # 再次健壮性检查
    #             normal_len = numpy.linalg.norm(normal)
    #             if normal_len < 1e-6:
    #                 # 极端情况下 fallback 还是零，直接跳过
    #                 continue
    #             normal = normal / normal_len
    #             centroid = (v1 + v2 + v3) / 3
    #             # 去重
    #             if not any(numpy.linalg.norm(centroid - s['site']) < 0.5 for s in hollow_sites):
    #                 hollow_sites.append({'site': centroid, 'normal': normal, 'type': 'Hollow'})
    #     return hollow_sites
    
    # def _find_hollow_sites(self, surface_atoms_coords: numpy.ndarray) -> list:
    #     self.logger.info("-> 使用 Alpha Shape 提取‘空腔’位点...")
    #     away_direction_vec = numpy.zeros(3)
    #     away_direction_vec[self.surface_axis] = 1.0 if self.place_on_bottom else -1.0
    #     if len(surface_atoms_coords) < 4:
    #         self.logger.warning("表面原子数不足，无法构建三维表面。")
    #         return []
    #     faces = None
    #     try:
    #         alpha = 5.0
    #         alpha_shape = alphashape.alphashape(surface_atoms_coords, alpha)
    #         faces = numpy.array(alpha_shape.triangles)
    #         # faces必须是二维(N,3)，否则回退到ConvexHull
    #         if faces.ndim != 2 or faces.shape[1] != 3:
    #             raise ValueError("Alpha shape faces格式异常，回退到ConvexHull")
    #     except Exception as e:
    #         self.logger.warning(f"Alpha shape failed或faces格式异常，回退到凸包。原因: {e}")
    #         hull = ConvexHull(surface_atoms_coords)
    #         faces = numpy.array(hull.simplices)
    #         if faces.ndim == 1 and len(faces) == 3:
    #             faces = faces.reshape(1, 3)
    #         if faces.ndim != 2 or faces.shape[1] != 3:
    #             self.logger.error(f"ConvexHull faces格式异常: {faces.shape if hasattr(faces, 'shape') else type(faces)}")
    #             return []
    #     if faces is None or faces.size == 0:
    #         self.logger.error("faces为空，无法生成空腔位点。")
    #         return []

    #     hollow_sites = []
    #     away_direction_vec = numpy.zeros(3)
    #     away_direction_vec[self.surface_axis] = -1.0 if self.place_on_bottom else 1.0

    #     for face in faces:
    #         v1, v2, v3 = surface_atoms_coords[face[0]], surface_atoms_coords[face[1]], surface_atoms_coords[face[2]]
    #         centroid = (v1 + v2 + v3) / 3
    #         normal = numpy.cross(v2 - v1, v3 - v1)
    #         norm_val = numpy.linalg.norm(normal)
    #         if norm_val < 1e-6:
    #           normal = away_direction_vec.copy()
    #         else:
    #           normal = normal / norm_val
    #           if numpy.dot(normal, away_direction_vec) < 0:
    #             normal = -normal
    #         if not any(numpy.linalg.norm(centroid - site['site']) < self.hollow_site_deduplication_distance for site in hollow_sites):
    #            hollow_sites.append({'site': centroid, 'normal': normal, 'type': 'Hollow'})
    #     self.logger.info(f"--> Alpha Shape后得到 {len(hollow_sites)} 个空腔位点。")
    #     return hollow_sites
   

    

    def _find_on_top_sites(self, slab: ase.Atoms, surface_atoms_indices: numpy.ndarray) -> list:
        self.logger.info(f"-> 计算“顶”位点 on '{self.on_top_target_atom}'...")
        sites_data, normal_vector = [], numpy.zeros(3)
        normal_vector[self.surface_axis] = -1.0 if self.place_on_bottom else 1.0
        for atom_index in surface_atoms_indices:
            atom = slab[atom_index]
            if atom.symbol == self.on_top_target_atom:
                sites_data.append({'site': atom.position, 'normal': normal_vector, 'type': 'On-Top'})
        self.logger.info(f"--> 找到了 {len(sites_data)} 个 '{self.on_top_target_atom}' 顶位点。")
        return sites_data

    def _place_and_optimize_adsorbate(self, slab: ase.Atoms, adsorbate: ase.Atoms, sites_data: list, e_slab: float, e_adsorbate: float) -> tuple[list, int]:
        self.logger.info(f"开始在 {len(sites_data)} 个位点上放置、旋转和筛选...")
        results_list, total_sites, skipped_sites = [], len(sites_data), 0
        chgnet_calculator = CHGNetCalculator()
        lj_calculator = LennardJones()

        for i, item in enumerate(sites_data):
            surface_site_cart, normal_cart, site_type = item['site'], item['normal'], item.get('type', 'Unknown')
            self.logger.info(f"--- 正在处理位点 {i+1}/{total_sites} ({site_type}) ---")

            target_pos = surface_site_cart + normal_cart * self.adsorption_height
            self.logger.info(f"    -> 目标位置: {target_pos} (法向: {normal_cart}, 吸附高度: {self.adsorption_height} Å)")
            adsorbate_to_place = self._place_adsorbate_at_site(adsorbate, target_pos)
            system = slab.copy()
            system.extend(adsorbate_to_place)

            adsorbate_indices = list(range(len(slab), len(system)))

            self.logger.info("    -> (使用LennardJones进行快速旋转优化...)")
            system.calc = lj_calculator
            system.constraints = FixAtoms(indices=range(len(slab)))
            
            if self.rotation_method:
             rotation_axes = self.generate_uniform_rotations(self.rotation_count)
             best_energy = float("inf")
             best_system = None
             best_axis = None
             best_angle = None

             adsorbate_com = system[adsorbate_indices].get_center_of_mass()
             for axis_dir in rotation_axes:
                axis_dir = axis_dir / numpy.linalg.norm(axis_dir)
                for angle in numpy.arange(0, 360, self.rotation_step):  # 多角度尝试
                    rot = Rotation.from_rotvec(numpy.radians(angle) * axis_dir)
                    trial_system = self._get_rotated_system(
                        rotation_obj=rot,
                        system=system,
                        adsorbate_indices=adsorbate_indices,
                        center_cart=adsorbate_com
                    )
                    trial_system.calc = lj_calculator
                    energy = trial_system.get_potential_energy()
                    if energy < best_energy:
                        best_energy = energy
                        best_system = trial_system
                        best_axis = axis_dir
                        best_angle = angle

             if best_system is None:
                self.logger.warning(f"    -> 位点 {i+1} 所有旋转均失败，跳过。")
                skipped_sites += 1
                continue

             dist_matrix = best_system.get_all_distances(mic=True)
             min_dist = numpy.min(dist_matrix[numpy.ix_(adsorbate_indices, range(len(slab)))])
             if min_dist < self.collision_threshold:
                self.logger.warning(f"    -> 位点 {i+1} 在最优旋转后仍发生碰撞 (最短距离: {min_dist:.2f} Å)，跳过。")
                skipped_sites += 1
                continue

             self.logger.info("    -> (使用CHGNet计算最终吸附能...)")
             best_system.calc = chgnet_calculator
             best_system_energy = best_system.get_potential_energy()
             best_system.constraints = []
             adsorption_energy = best_system_energy - (e_slab + e_adsorbate)
             self.logger.info(f"    -> 位点{i+1}优化完成，最佳方向: {best_axis}, 最佳角度: {best_angle}°, 能量: {best_system_energy:.4f} eV,最小间距: {min_dist:.2f} Å")
             filename = f"02_adsorbed_site_{i+1}_{site_type}.cif"
             ase.io.write(os.path.join(self.output_folder, filename), best_system)
             self.logger.info(f"    -> 已保存至 {filename}")
             results_list.append({
                'system': best_system,
                'adsorption_energy': adsorption_energy,
                'site_type': site_type,
                'surface_site_coordinates': surface_site_cart.tolist(),
                'adsorbate_com_coordinates': best_system[adsorbate_indices].get_center_of_mass().tolist(),
                'site_index': i + 1
             })
            else:
                rotation_center_cart = system[adsorbate_indices].get_center_of_mass()
                energy_func_lj = lambda angle: self._get_rotation_energy_normal(angle, system, adsorbate_indices, rotation_center_cart, normal_cart)
                opt_result = minimize_scalar(energy_func_lj, bounds=(0, 360), method='bounded')
                
                final_system = self._get_rotated_system_normal(opt_result.x, system, adsorbate_indices, rotation_center_cart, normal_cart)
                
                min_dist = numpy.min(final_system.get_all_distances(mic=True)[adsorbate_indices][:,:len(slab)])
                if min_dist < self.collision_threshold:
                    self.logger.warning(f"    -> 位点 {i+1} 在最优旋转后仍发生碰撞 (最短距离: {min_dist:.2f} Å)，跳过。")
                    skipped_sites += 1
                    continue
                final_system.calc = chgnet_calculator
                e_total = final_system.get_potential_energy()
                adsorption_energy = e_total - (e_slab + e_adsorbate)
                
                final_system.calc = None
                final_system.constraints = []
                self.logger.info(f"    -> 位点 {i+1} 优化完成。最优角度: {opt_result.x:.2f}°, 吸附能: {adsorption_energy:.4f} eV。")

            
                filename = f"02_adsorbed_site_{i+1}_{site_type}.cif"
                ase.io.write(os.path.join(self.output_folder, filename), final_system)
            
                self.logger.info(f"    -> 已保存至 {filename}")

                results_list.append({
                    'system': final_system,
                    'adsorption_energy': adsorption_energy,
                    'site_type': site_type,
                    'surface_site_coordinates': surface_site_cart.tolist(),
                    'adsorbate_com_coordinates': final_system[adsorbate_indices].get_center_of_mass().tolist(),
                    'site_index': i + 1
                })
            

        self.logger.info("-------------------- 优化流程总结 --------------------")
        self.logger.info(f"总共尝试位点数: {total_sites}")
        self.logger.info(f"因碰撞而跳过数: {skipped_sites}")
        self.logger.info(f"最终成功生成数: {len(results_list)}")
        self.logger.info("----------------------------------------------------")
        return results_list, total_sites
    
    def _create_json_for_frontend_visualization(self, slab: ase.Atoms, results: list):
        self.logger.info("正在创建用于前端可视化的 'adsorption_sites.json'...")
        output_data = {
            "cell": slab.cell.tolist(),
            "sites": []
        }
        for result in results:
            output_data["sites"].append({
                "coords": result['surface_site_coordinates'],
                "energy": float(round(result['adsorption_energy'], 4))
            })
        filename = os.path.join(self.output_folder, "adsorption_sites.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"成功将可视化数据写入到: {filename}")
            return output_data
        except Exception as e:
            self.logger.error(f"写入 'adsorption_sites.json' 时发生错误: {e}", exc_info=True)
            return None

    def _visualize_final_results(self, slab: ase.Atoms, results: list):
        self.logger.info("生成包含所有*优化后*吸附物的最终可视化文件...")
        final_viz_system = slab.copy()
        num_slab_atoms = len(slab)
        self.logger.info("--- 最终优化构型汇总 (吸附能越负越稳定) ---")
        sorted_results = sorted(results, key=lambda x: x['adsorption_energy'])
        for result in sorted_results:
            final_system = result['system']
            adsorbate_atoms = final_system[num_slab_atoms:]
            final_viz_system.extend(adsorbate_atoms)
            symbol = ['Xe', 'Kr', 'Ar', 'Ne', 'He'][(result['site_index']-1) % 5]
            final_viz_system.append(ase.Atom(symbol, position=result['surface_site_coordinates']))
            self.logger.info(f"  位点 {result['site_index']} ({result['site_type']}): "
                        f"吸附能 = {result['adsorption_energy']:.4f} eV. "
                        f"(可视化标记: {symbol})")
        filename = "00_FINAL_all_optimized_sites.cif"
        ase.io.write(os.path.join(self.output_folder, filename), final_viz_system)
        self.logger.info(f"包含所有最终构型的可视化文件已生成: {filename}")
        
    def _place_adsorbate_at_site(self, adsorbate: ase.Atoms, target_pos_cart: numpy.ndarray) -> ase.Atoms:
        placed_adsorbate = adsorbate.copy()
        current_com = placed_adsorbate.get_center_of_mass()
        translation_vector = target_pos_cart - current_com
        placed_adsorbate.translate(translation_vector)
        return placed_adsorbate
        
    def _get_rotated_system_normal(self, angle_deg, system, adsorbate_indices, center_cart, axis_cart):
       rotated_system = system.copy()
       adsorbate_part = system[adsorbate_indices]
       adsorbate_part.rotate(angle_deg, v=axis_cart, center=center_cart)
       rotated_system.positions[adsorbate_indices] = adsorbate_part.positions
       return rotated_system
   
    def _get_rotation_energy_normal(self, angle_deg, system, adsorbate_indices, center_cart, axis_cart):
        temp_system = self._get_rotated_system_normal(angle_deg, system, adsorbate_indices, center_cart, axis_cart)
        temp_system.calc = system.calc
        return temp_system.get_potential_energy()
    """法向旋转版本"""

    def _get_rotated_system(self, rotation_obj, system, adsorbate_indices, center_cart):
     rotated_system = system.copy()
     adsorbate_part = rotated_system[adsorbate_indices]
     pos = adsorbate_part.get_positions()
     translated = pos - center_cart
     rotated = rotation_obj.apply(translated) + center_cart
     adsorbate_part.set_positions(rotated)
     rotated_system.positions[adsorbate_indices] = adsorbate_part.positions
     return rotated_system


    def _get_rotation_energy(self, angle_deg, system, adsorbate_indices, center_cart, axis_cart):
        temp_system = self._get_rotated_system(angle_deg, system, adsorbate_indices, center_cart, axis_cart)
        temp_system.calc = system.calc 
        return temp_system.get_potential_energy()
    
    def generate_uniform_rotations(self, num_rotations):
     indices = numpy.arange(0, num_rotations, dtype=float) + 0.5
     phi = numpy.arccos(1 - 2 * indices / num_rotations)
     theta = numpy.pi * (1 + 5 ** 0.5) * indices

     directions = numpy.stack([
        numpy.sin(phi) * numpy.cos(theta),
        numpy.sin(phi) * numpy.sin(theta),
        numpy.cos(phi)
    ], axis=-1)

     return directions  # 返回向量

# --- NEW SCRIPT EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Surface Adsorption Workflow")
    parser.add_argument('--substrate', required=True, help='Path to substrate CIF file')
    parser.add_argument('--adsorbate', required=True, help='Path to adsorbate CIF file')
    parser.add_argument('--output_folder', default='adsorption_results', help='Folder to save results')
    # Add all other parameters from the form
    parser.add_argument('--adsorption_height', type=float, default=2.0)
    parser.add_argument('--vacuum_thickness', type=float, default=20.0)
    parser.add_argument('--surface_search_depth', type=float, default=3.5)
    parser.add_argument('--hollow_sites_enabled', action='store_true')
    parser.add_argument('--on_top_sites_enabled', action='store_true')
    parser.add_argument('--on_top_target_atom', type=str, default='O')
    parser.add_argument('--surface_axis', type=int, default=2)
    parser.add_argument('--collision_threshold', type=float, default=1.2)
    parser.add_argument('--knn_neighbors', type=int, default=2)
    parser.add_argument('--hollow_site_deduplication_distance', type=float, default=1.5)
    parser.add_argument('--place_on_bottom', action='store_true')
    parser.add_argument('--rotation_count', type=int, default=50, help='Number of rotation axes to sample')
    parser.add_argument('--rotation_step', type=float, default=30, help='Rotation step in degrees')
    parser.add_argument('--rotation_method', action='store_true', help='Enable sphere rotation method')

    args = parser.parse_args()

    # Configure file logging inside the script
    file_handler = logging.FileHandler(os.path.join(args.output_folder, "workflow.log"), encoding="utf-8", mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Convert args to a dictionary to pass to the workflow
    workflow_params = vars(args)
    
    workflow = SurfaceAdsorptionWorkflow(**workflow_params)
    workflow.run(substrate_path=args.substrate, adsorbate_path=args.adsorbate)