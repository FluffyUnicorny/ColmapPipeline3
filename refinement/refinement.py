import numpy as np

# ถ้าคุณมี ICP อยู่ไฟล์ไหน ให้ import ตรงนี้
# from alignment.icp import run_icp
# from utils.geometry import project_points

def remove_outliers_statistical(points, z_thresh=3.0):
    """
    Simple statistical outlier removal based on distance to centroid
    """
    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)

    mean = np.mean(dists)
    std = np.std(dists)

    mask = dists < mean + z_thresh * std
    return points[mask], mask


def reprojection_filter(points_3d, reproj_errors, thresh=3.0):
    """
    Filter points using reprojection error (if available)
    """
    mask = reproj_errors < thresh
    return points_3d[mask], mask


def refine_model(
    points_3d,
    poses,
    reproj_errors=None,
    use_icp=True,
    use_ba=False,
    voxel_size=0.05,
):
    """
    Refinement stage of the pipeline

    Parameters
    ----------
    points_3d : (N, 3) ndarray
        Aligned 3D points
    poses : dict or ndarray
        Camera poses after alignment (PnP / initial ICP)
    reproj_errors : ndarray, optional
        Reprojection errors per 3D point
    use_icp : bool
        Whether to apply ICP refinement
    use_ba : bool
        Whether to apply bundle adjustment (usually from COLMAP)
    voxel_size : float
        Downsampling size (optional)

    Returns
    -------
    refined_points : ndarray
        Refined 3D points
    refined_poses : same type as poses
        Refined camera poses
    """

    refined_points = points_3d.copy()
    refined_poses = poses

    # ----------------------------------
    # 1. Outlier removal
    # ----------------------------------
    refined_points, inlier_mask = remove_outliers_statistical(
        refined_points, z_thresh=3.0
    )

    # ----------------------------------
    # 2. Reprojection error filtering
    # ----------------------------------
    if reproj_errors is not None:
        reproj_errors = reproj_errors[inlier_mask]
        refined_points, reproj_mask = reprojection_filter(
            refined_points, reproj_errors, thresh=3.0
        )

    # ----------------------------------
    # 3. ICP refinement (reuse existing code)
    # ----------------------------------
    if use_icp:
        # ตัวอย่าง:
        # refined_points, refined_poses = run_icp(
        #     refined_points,
        #     refined_poses,
        #     voxel_size=voxel_size
        # )
        pass

    # ----------------------------------
    # 4. Bundle Adjustment (optional)
    # ----------------------------------
    if use_ba:
        # ปกติใช้ BA จาก COLMAP
        # หรือเรียก optimizer ที่มีอยู่แล้ว
        pass

    return refined_points, refined_poses
