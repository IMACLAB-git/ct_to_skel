"""Coordinate frames.

* CT world: DICOM LPS, millimetres  (x = patient left, y = posterior, z = superior)
* SKEL   : SMPL convention, metres  (x = patient left, y = superior, z = anterior)

For a supine patient, the CT simply describes the body "standing" in the SKEL
frame once the axes are permuted; no gravity-dependent assumption is made.
"""
from __future__ import annotations

import numpy as np

# LPS -> SKEL axis permutation (rotation matrix, det = +1)
R_LPS_TO_SKEL = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]])
MM_TO_M = 1e-3


class FrameTransform:
    """Affine map between CT (LPS mm) and SKEL frame (m), centred on ``center_mm``."""

    def __init__(self, center_mm):
        self.center_mm = np.asarray(center_mm, dtype=np.float64).reshape(3)

    # LPS mm -> SKEL m
    def lps_to_skel(self, pts_mm: np.ndarray) -> np.ndarray:
        p = (np.asarray(pts_mm, dtype=np.float64) - self.center_mm) @ R_LPS_TO_SKEL.T
        return p * MM_TO_M

    # SKEL m -> LPS mm
    def skel_to_lps(self, pts_m: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_m, dtype=np.float64) / MM_TO_M
        return p @ R_LPS_TO_SKEL + self.center_mm

    def rotate_lps_to_skel(self, vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64) @ R_LPS_TO_SKEL.T

    def matrix_lps_to_skel(self) -> np.ndarray:
        A = np.eye(4)
        A[:3, :3] = R_LPS_TO_SKEL * MM_TO_M
        A[:3, 3] = -R_LPS_TO_SKEL @ self.center_mm * MM_TO_M
        return A

    def matrix_skel_to_lps(self) -> np.ndarray:
        return np.linalg.inv(self.matrix_lps_to_skel())

    def to_dict(self) -> dict:
        return {
            "description": "SKEL frame = R * (LPS_mm - center_mm) * 1e-3; export STLs are SKEL frame in mm",
            "center_mm": self.center_mm.tolist(),
            "R_lps_to_skel": R_LPS_TO_SKEL.tolist(),
            "matrix_lps_to_skel_m": self.matrix_lps_to_skel().tolist(),
            "matrix_skelmm_to_lps": np.linalg.inv(self._matrix_lps_to_skel_mm()).tolist(),
        }

    def _matrix_lps_to_skel_mm(self) -> np.ndarray:
        A = np.eye(4)
        A[:3, :3] = R_LPS_TO_SKEL
        A[:3, 3] = -R_LPS_TO_SKEL @ self.center_mm
        return A
