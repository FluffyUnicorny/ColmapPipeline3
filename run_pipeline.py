# run_pipeline.py
from pathlib import Path
import subprocess
import sys
import numpy as np
import tempfile

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_NAME = "dataset_pipes"

ROOT = Path(__file__).resolve().parent

# -----------------------------
# DATA PATHS
# -----------------------------
IMG_DIR = ROOT / f"data/{DATASET_NAME}/images"
COLMAP_WS = ROOT / f"data/{DATASET_NAME}/colmap"
COLMAP_SPARSE = COLMAP_WS / "sparse/0"

GT_2D_DIR = ROOT / f"data/{DATASET_NAME}/gt_2d"
GT_3D_PLY = ROOT / f"data/{DATASET_NAME}/ground_truth_eval/scan1.ply"

EVAL_2D_DIR = ROOT / "evaluation/evaluation_2d"
EVAL_3D_DIR = ROOT / "evaluation/evaluation_3d"

for d in [COLMAP_WS, EVAL_2D_DIR, EVAL_3D_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# IMPORTS
# -----------------------------
from shared.colmap_io import load_colmap_sparse
from evaluation.evaluate import evaluate_reprojection
from evaluation.eval_align_and_localize import evaluate_alignment

# <<< NEW >>>
from refinement.refinement import refine_model


# -----------------------------
# PIPELINE STEPS
# -----------------------------
def run_colmap():
    print("\n▶ STEP 1: Running COLMAP SfM")

    cmd = [
        "colmap", "automatic_reconstructor",
        "--image_path", str(IMG_DIR),
        "--workspace_path", str(COLMAP_WS),
        "--dense", "0"
    ]

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("✅ COLMAP SfM finished")
    except Exception as e:
        print(f"[WARN] COLMAP failed or already exists: {e}")
        print("       Using existing COLMAP output")


# <<< NEW >>>
def run_refinement():
    print("\n▶ STEP 2: Refinement (ICP / filtering)")

    pts3d, poses = load_colmap_sparse(COLMAP_SPARSE)

    if pts3d.shape[0] == 0 or len(poses) == 0:
        print("[SKIP] No data for refinement")
        return pts3d, poses

    refined_pts, refined_poses = refine_model(
        points_3d=pts3d,
        poses=poses,
        use_icp=True,
        use_ba=False
    )

    print(f"✅ Refinement done: {pts3d.shape[0]} → {refined_pts.shape[0]} points")
    return refined_pts, refined_poses


def run_eval_2d(pts3d, poses):
    print("\n▶ STEP 3: 2D Reprojection Evaluation")

    if not GT_2D_DIR.exists():
        print("[SKIP] No 2D GT found")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    pts_file = tmp_dir / "points3d.npy"
    cam_file = tmp_dir / "camera_poses.npy"

    np.save(pts_file, pts3d)
    np.save(
        cam_file,
        {k: {"rvec": v["R"], "tvec": v["t"]} for k, v in poses.items()}
    )

    rms, used = evaluate_reprojection(
        pts_file, cam_file, IMG_DIR, GT_2D_DIR
    )

    print(f"[RESULT] RMS reprojection error: {rms:.2f}px (images={used})")


def run_distance_analysis(pts3d, poses):
    print("\n▶ STEP 3.5: Camera-to-Object Distance Analysis")

    if pts3d.shape[0] == 0 or len(poses) == 0:
        print("[SKIP] No data")
        return

    object_center = pts3d.mean(axis=0)
    cam_positions = np.array([v["t"] for v in poses.values()])
    distances = np.linalg.norm(cam_positions - object_center, axis=1)

    print("Camera-to-object distance (relative scale):")
    print(f"  min  : {distances.min():.2f}")
    print(f"  mean : {distances.mean():.2f}")
    print(f"  max  : {distances.max():.2f}")


def run_eval_3d():
    print("\n▶ STEP 4: 3D Alignment Evaluation")

    evaluate_alignment(
        estimated_points_file=COLMAP_SPARSE,
        gt_ply_file=GT_3D_PLY,
        out_dir=EVAL_3D_DIR
    )


# -----------------------------
# RUN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print(f"🚀 VisionPipeline started with dataset: {DATASET_NAME}")

    run_colmap()

    refined_pts, refined_poses = run_refinement()   # <<< NEW >>>

    run_eval_2d(refined_pts, refined_poses)
    run_distance_analysis(refined_pts, refined_poses)
    run_eval_3d()

    print("\n🎉 Pipeline finished successfully")
