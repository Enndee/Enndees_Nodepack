"""QC: compare a Camerapath.txt (planned path) with sparse/0/images.txt (SfM poses).

Run after the GLOMAP tracker to measure how faithfully a generated video followed
the planned camera path - and to decide whether the Path Pose Dataset node's
ground-truth extrinsics are safe to train on for this clip.

    python compare_path_vs_colmap.py <dataset_folder> [stride_hint]

The dataset folder must contain ``Camerapath.txt`` and ``sparse/0/images.txt``
(the 20261014__23_43_33-style Lichtfeld export). Reports the best-fit similarity
(Umeyama) between the planned keys and the solved poses, the assumed frame<->key
mapping, and the per-frame translation / view-direction / up-vector deviations.

Reference numbers (20261014__23_43_33_upscaled2x_RTXUpscale2xMP_00001):
~5-7 deg rotation and ~6-11 % of the orbit radius position deviation - the clip
only approximately follows the path, which is exactly what this tool measures.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def qrot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def umeyama(X, Y):
    """Optimal similarity (scale + rotation + translation) mapping X onto Y."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    var_x = (Xc**2).sum() / len(X)
    H = Xc.T @ Yc / len(X)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return np.trace(np.diag(D) @ S) / var_x, R, my - (np.trace(np.diag(D) @ S) / var_x) * (R @ mx)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    doc = json.loads((root / "Camerapath.txt").read_text(encoding="utf-8"))
    P = np.array([key["pos"] for key in doc["path"]], dtype=np.float64)
    L = np.array([key["look"] for key in doc["path"]], dtype=np.float64)
    n_path = len(P)

    qvecs, tvecs = [], []
    for line in (root / "sparse" / "0" / "images.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or len(line.split()) != 10:
            continue
        fields = line.split()
        qvecs.append([float(v) for v in fields[1:5]])
        tvecs.append([float(v) for v in fields[5:8]])
    R_w2c = np.array([qrot(q) for q in qvecs])
    C = np.array([-R.T @ t for R, t in zip(R_w2c, tvecs)])
    fwd_c = np.array([R.T @ np.array([0, 0, 1.0]) for R in R_w2c])
    up_c = np.array([R.T @ np.array([0, -1.0, 0]) for R in R_w2c])
    n_img = len(C)
    fwd_p = L - P
    fwd_p /= np.linalg.norm(fwd_p, axis=1, keepdims=True)

    best = None
    for stride in np.linspace(3.4, 4.6, 25):
        for offset in range(-4, 5):
            idx = np.clip(np.round(np.arange(n_img) * stride + offset).astype(int), 0, n_path - 1)
            if len(np.unique(idx)) < n_img:
                continue
            s, R, t = umeyama(P[idx], C)
            res = np.linalg.norm(C - ((s * (R @ P[idx].T)).T + t), axis=1)
            rms = float(np.sqrt((res**2).mean()))
            if best is None or rms < best[0]:
                best = (rms, stride, offset, idx, s, R, t, res)
    rms, stride, offset, idx, s, R_aln, t_aln, res = best
    orbit = float(np.linalg.norm(P[0] - L[0]))
    print(f"frames: path={n_path} images={n_img}  mapping: image i -> key round(i*{stride:.2f}{offset:+d})")
    print(f"scale: {s:.4f} colmap-units per path-unit")
    print(f"position residual: RMS {rms:.4f} ({100*rms/(orbit*s):.1f}% of orbit radius), "
          f"max {res.max():.4f} at image {int(np.argmax(res))+1}")
    ang_f, ang_u = [], []
    for i in range(n_img):
        fp = R_aln @ fwd_p[idx[i]]
        ang_f.append(math.degrees(math.acos(np.clip(np.dot(fp, fwd_c[i]), -1, 1))))
        best_up = None
        for sgn in (+1.0, -1.0):
            g = np.array([0.0, sgn, 0.0])
            up = g - np.dot(g, fwd_p[idx[i]]) * fwd_p[idx[i]]
            au = math.degrees(math.acos(np.clip(np.dot(R_aln @ (up / np.linalg.norm(up)), up_c[i]), -1, 1)))
            best_up = au if best_up is None or au < best_up else best_up
        ang_u.append(best_up)
    print(f"view-direction error: mean {np.mean(ang_f):.2f} deg, max {np.max(ang_f):.2f} deg")
    print(f"horizon (up) error:   mean {np.mean(ang_u):.2f} deg, max {np.max(ang_u):.2f} deg")
    print("-> under ~1 deg / ~1%: safe to train on the planned path poses; above: keep the SfM poses")


if __name__ == "__main__":
    main()
