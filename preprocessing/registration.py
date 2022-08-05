import copy
import logging
import numpy as np
import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size, normals_nn=30, fpfh_nn=100):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=normals_nn)
        )
    
    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=fpfh_nn)
        )
    return pcd_down, pcd_fpfh


def prepare_dataset(pcd_master, pcd_sub, voxel_size, normals_nn=40, fpfh_nn=40):
    source = copy.deepcopy(pcd_sub)
    target = copy.deepcopy(pcd_master)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, normals_nn, fpfh_nn)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, normals_nn, fpfh_nn)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    pcd_master: o3d.geometry.PointCloud, 
    pcd_sub: o3d.geometry.PointCloud,
    voxel_size: int = 35,
    ransac_n_trials: int = 10
    ) -> None:
    """
    Apply a global registration between pcd_master and pcd_sub, 
    where the result is a 4x4 transformation matrix that transforms
    all points from pcd_sub to the coordinate frame of pcd_master
    """
    best_fitness = 0
    ransac_transformation = None
    for _ in range(ransac_n_trials):
        (source, target, source_down, target_down, 
         source_fpfh, target_fpfh) = prepare_dataset(pcd_master, pcd_sub, voxel_size)

        distance_threshold = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, 
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3, 
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
                o3d.pipelines.registration.RANSACConvergenceCriteria(250000, 0.999)
            )

        if best_fitness < result_ransac.fitness:
            best_fitness = result_ransac.fitness
            ransac_transformation = result_ransac.transformation
    return ransac_transformation


def execute_point_to_plane_registration(
    pcd_master: o3d.geometry.PointCloud, 
    pcd_sub: o3d.geometry.PointCloud, 
    initial_transformation: np.ndarray, 
    voxel_size: int = 35
    ) -> np.ndarray:
    print('Starting local registration refinement')

    source = copy.deepcopy(pcd_master)
    target = copy.deepcopy(pcd_sub)
    threshold = 100

    _, _, source_down, target_down, _, _ = prepare_dataset(source, target, voxel_size)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_down, 
        target_down, 
        threshold, 
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
        
    return reg_p2l.transformation

def execute_colored_ICP_registration(pcd_master, pcd_sub, initial_transformation):
    logging.info('Starting local registration refinement')

    source = copy.deepcopy(pcd_master)
    target = copy.deepcopy(pcd_sub)
    
    voxel_radius = [80, 40, 20]
    max_iter = [50, 30, 14]
    for scale in range(len(max_iter)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
    
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
    return result_icp.transformation
