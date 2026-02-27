"""projection.py — LDraw↔OpenSCAD coordinate transforms and camera projection."""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Coordinate transform: LDraw ↔ OpenSCAD
# ---------------------------------------------------------------------------
# LDraw:  X right, Y down, Z toward viewer  (units: LDU)
# LEGO.scad/OpenSCAD: X right, Y forward, Z up  (units: mm)
# Mapping: os_X = ld_X * 0.4,  os_Y = -ld_Z * 0.4,  os_Z = -ld_Y * 0.4
_T = np.array([[1, 0, 0],   # os_X  = ld_X
               [0, 0,-1],   # os_Y  = -ld_Z
               [0,-1, 0]],  # os_Z  = -ld_Y
              dtype=float)

LDU_TO_MM = 0.4

def ldraw_to_os(pos_ld: np.ndarray, rot_ld: np.ndarray) -> np.ndarray:
    """Build a 4×4 OpenSCAD transform from a LDraw position+rotation."""
    R_os = _T @ rot_ld @ _T
    t_os = _T @ pos_ld * LDU_TO_MM
    m = np.eye(4)
    m[:3, :3] = R_os
    m[:3,  3] = t_os
    return m

# ---------------------------------------------------------------------------
# Camera parameters (OpenSCAD)
# ---------------------------------------------------------------------------
# True isometric: camera along (1,1,1)/√3 ↔ rx = arccos(1/√3) ≈ 54.74°, rz = 45°.
# All three world axes project with equal foreshortening; horizontal axes
# appear at exactly ±30° on screen.
CAMERA_RX = math.degrees(math.acos(1 / math.sqrt(3)))  # ≈ 54.74°
CAMERA_RZ = 45.0   # degrees spin (around Z)
CAMERA_D  = 300.0  # camera distance (mm) — controls scale
IMG_PX    = 800    # render each piece into a square IMG_PX × IMG_PX PNG

# OpenSCAD ortho scale: viewport covers 2*D*tan(fov/2) mm → IMG_PX pixels.
_OPENSCAD_FOV_DEG = 22.5
PX_PER_MM = IMG_PX / (2 * CAMERA_D * math.tan(math.radians(_OPENSCAD_FOV_DEG / 2)))

# ---------------------------------------------------------------------------
# Camera matrix and projection
# ---------------------------------------------------------------------------
def _cam_matrix(rx_deg: float, rz_deg: float) -> np.ndarray:
    """R_cam = Rx(rx) @ Rz(rz) — maps OpenSCAD world → camera space."""
    rx = np.radians(rx_deg)
    rz = np.radians(rz_deg)
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(rx), -np.sin(rx) ],
                   [0, np.sin(rx),  np.cos(rx) ]])
    Rz = np.array([[ np.cos(rz), -np.sin(rz), 0],
                   [ np.sin(rz),  np.cos(rz), 0],
                   [ 0,           0,          1]])
    return Rx @ Rz

_R_CAM = _cam_matrix(-CAMERA_RX, -CAMERA_RZ)

def project_ldraw(pos_ld: np.ndarray) -> tuple[float, float, float]:
    """Project a LDraw world position to (screen_x, screen_y, depth).

    screen_y increases downward (SVG convention).
    depth increases toward the camera (larger = in front).
    """
    p_os = _T @ pos_ld * LDU_TO_MM
    cam  = _R_CAM @ p_os
    return float(cam[0]), float(-cam[1]), float(cam[2])
