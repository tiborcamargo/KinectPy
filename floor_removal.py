import glob
import pathlib
import numpy as np
import open3d as o3d


def pick_points(pcd: o3d.geometry.PointCloud):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def equation_plane(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    print("equation of plane is ", a, "x +", b, "y +", c, "z +", d, "= 0.")
    return a, b, c, d


def pcd_above_plane(a, b, c, d, pcd):
    above_or_below = []
    pcd_points = np.asarray(pcd.points)
    for point in pcd_points:
        plane = a*point[0] + b*point[1] + c*point[2] + d
        if plane >= 0:
            color = 1
        else:
            color = 0
        above_or_below.append(color)
    above_or_below = np.array(above_or_below)
    above_cloud = pcd.select_by_index(np.argwhere(above_or_below == 0))
    return above_cloud


if __name__ == '__main__':
    pcds = glob.glob('E:/Extracted_data/*/*/*/master_1/filtered_and_registered_pointclouds/*.pcd')

    for i, pcd_fp in enumerate(pcds):
        if i % 100 == 0:
            print(i)

        pcd = o3d.io.read_point_cloud(pcd_fp)

        # Find where the floor is
        pcd_points = np.asarray(pcd.points)
        idx_lower = np.argwhere(pcd_points[:, 1] >= pcd_points[:, 1].max() - 200)
        idx_upper = np.argwhere(pcd_points[:, 1] < pcd_points[:, 1].max() - 200)

        # Remove the floor and further apply statistical filter
        floor = pcd.select_by_index(idx_lower)
        plane_model, inliers = floor.segment_plane(distance_threshold=30, ransac_n=30, num_iterations=2000)
        outlier_cloud = floor.select_by_index(inliers, invert=True) # not removed points (feet, and so on)
        filtered_pcd = outlier_cloud + pcd.select_by_index(idx_upper)
        filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(50, 0.30)

        # Save - If folder does not exist, then create
        dst = pcd_fp.replace('filtered_and_registered_pointclouds', 'floor_filter_and_registered_pointclouds')
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(dst, filtered_pcd)
