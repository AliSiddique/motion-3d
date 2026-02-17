"""
SMPL → Mixamo/RPM skeleton retargeting.

Takes raw SMPL-H motion data (201-dim per frame) and converts it to
Mixamo-compatible bone rotations suitable for RPM avatars.

Output formats: BVH (text-based, universal) and optionally FBX.
"""

import json
import logging
import struct
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ── SMPL → Mixamo bone mapping ──────────────────────────────────────
# From the animation-retargeting-research plan.
# Maps SMPL joint index → Mixamo/RPM bone name.
# Only the 20 joints that have a direct 1:1 mapping.

SMPL_TO_MIXAMO = {
    0: "Hips",           # pelvis → Hips (root)
    3: "Spine",          # spine1 → Spine
    6: "Spine1",         # spine2 → Spine1 (naming offset!)
    9: "Spine2",         # spine3 → Spine2 (naming offset!)
    12: "Neck",          # neck
    15: "Head",          # head
    13: "LeftShoulder",  # left_collar → LeftShoulder (clavicle)
    14: "RightShoulder", # right_collar → RightShoulder (clavicle)
    16: "LeftArm",       # left_shoulder → LeftArm (naming mismatch!)
    17: "RightArm",      # right_shoulder → RightArm (naming mismatch!)
    18: "LeftForeArm",   # left_elbow
    19: "RightForeArm",  # right_elbow
    20: "LeftHand",      # left_wrist → LeftHand
    21: "RightHand",     # right_wrist → RightHand
    1: "LeftUpLeg",      # left_hip → LeftUpLeg
    2: "RightUpLeg",     # right_hip → RightUpLeg
    4: "LeftLeg",        # left_knee
    5: "RightLeg",       # right_knee
    7: "LeftFoot",       # left_ankle
    8: "RightFoot",      # right_ankle
}

# SMPL joint indices to skip (no RPM equivalent)
SMPL_SKIP_JOINTS = {10, 11, 22, 23}  # foot tips, hand roots

# SMPL joint names for reference
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Mixamo skeleton hierarchy (parent → children)
MIXAMO_HIERARCHY = {
    "Hips": ["Spine", "LeftUpLeg", "RightUpLeg"],
    "Spine": ["Spine1"],
    "Spine1": ["Spine2"],
    "Spine2": ["Neck", "LeftShoulder", "RightShoulder"],
    "Neck": ["Head"],
    "Head": [],
    "LeftShoulder": ["LeftArm"],
    "LeftArm": ["LeftForeArm"],
    "LeftForeArm": ["LeftHand"],
    "LeftHand": [],
    "RightShoulder": ["RightArm"],
    "RightArm": ["RightForeArm"],
    "RightForeArm": ["RightHand"],
    "RightHand": [],
    "LeftUpLeg": ["LeftLeg"],
    "LeftLeg": ["LeftFoot"],
    "LeftFoot": [],
    "RightUpLeg": ["RightLeg"],
    "RightLeg": ["RightFoot"],
    "RightFoot": [],
}

# Mixamo parent lookup (bone → parent bone name, None for root)
MIXAMO_PARENT: dict[str, str | None] = {"Hips": None}
for _parent, _children in MIXAMO_HIERARCHY.items():
    for _child in _children:
        MIXAMO_PARENT[_child] = _parent

# BVH channel order for each bone
BVH_CHANNELS = ["Zrotation", "Xrotation", "Yrotation"]

# ── RPM rest pose rotations ─────────────────────────────────────────
# ReadyPlayerMe (Wolf3D) skeleton has non-identity rest rotations that
# define bone orientations in T-pose. SMPL outputs local rotations
# relative to an identity rest pose. To get the same joint angle on RPM:
#   joint_angle = R_rpm @ R_rest^(-1) = R_smpl  =>  R_rpm = R_smpl @ R_rest
#
# Format: (x, y, z, w) quaternion — glTF convention, extracted from an
# actual RPM avatar GLB.  These are consistent across all RPM avatars
# (only bone lengths/translations vary).

RPM_REST_ROTATIONS_XYZW = {
    "Hips":           (-0.0300, -0.0000,  0.0000,  0.9996),
    "Spine":          (-0.0138,  0.0000,  0.0000,  0.9999),
    "Spine1":         (-0.0364,  0.0000,  0.0000,  0.9993),
    "Spine2":         ( 0.0636, -0.0000, -0.0000,  0.9980),
    "Neck":           ( 0.1396,  0.0000, -0.0000,  0.9902),
    "Head":           (-0.0534,  0.0000,  0.0000,  0.9986),
    "LeftShoulder":   ( 0.4196,  0.5679, -0.5267,  0.4732),
    "LeftArm":        ( 0.0445,  0.0339,  0.1019,  0.9932),
    "LeftForeArm":    ( 0.0172, -0.0053,  0.0006,  0.9998),
    "LeftHand":       (-0.0596, -0.0886, -0.0328,  0.9937),
    "RightShoulder":  ( 0.4196, -0.5679,  0.5267,  0.4732),
    "RightArm":       ( 0.0445, -0.0339, -0.1019,  0.9932),
    "RightForeArm":   ( 0.0172,  0.0053, -0.0006,  0.9998),
    "RightHand":      (-0.0596,  0.0886,  0.0328,  0.9937),
    "LeftUpLeg":      ( 0.0009,  0.0295, -0.9996,  0.0004),
    "LeftLeg":        (-0.0349,  0.0003, -0.0010,  0.9994),
    "LeftFoot":       ( 0.4770, -0.0232, -0.0146,  0.8785),
    "RightUpLeg":     ( 0.0010, -0.0295,  0.9996,  0.0004),
    "RightLeg":       (-0.0349, -0.0003,  0.0010,  0.9994),
    "RightFoot":      ( 0.4770,  0.0232,  0.0146,  0.8784),
}

# ── GLB rest pose extraction ───────────────────────────────────────

# Bones we look for when extracting rest poses from custom GLBs
_BONES_OF_INTEREST = set(MIXAMO_HIERARCHY.keys())


def extract_rest_from_glb(glb_data: bytes) -> dict[str, tuple[float, float, float, float]]:
    """Extract bone rest quaternions from a GLB file.

    Parses the binary GLB, finds bone nodes by name, and returns their
    rest rotation quaternions in (x, y, z, w) glTF format.

    Strips common prefixes (mixamorig:, mixamorig) from bone names so
    this works with both RPM avatars and raw Mixamo exports.

    Args:
        glb_data: Raw GLB file bytes.

    Returns:
        Dict mapping bone name → (x, y, z, w) rest quaternion.
    """
    # Parse GLB header
    magic, version, total_len = struct.unpack_from("<III", glb_data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file (magic={magic:#x})")

    # Chunk 0: JSON
    json_len, json_type = struct.unpack_from("<II", glb_data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("First chunk is not JSON")
    gltf = json.loads(glb_data[20 : 20 + json_len])

    # Extract bone quaternions
    result = {}
    for node in gltf.get("nodes", []):
        name = node.get("name", "")
        clean = name.replace("mixamorig:", "").replace("mixamorig", "")
        if clean in _BONES_OF_INTEREST:
            rot = node.get("rotation", [0, 0, 0, 1])  # glTF default = identity
            result[clean] = tuple(rot)  # (x, y, z, w)

    if not result:
        logger.warning("No Mixamo bones found in GLB — using RPM defaults")
        return RPM_REST_ROTATIONS_XYZW

    # Fill any missing bones from RPM defaults so the full skeleton is covered
    for bone in _BONES_OF_INTEREST:
        if bone not in result and bone in RPM_REST_ROTATIONS_XYZW:
            result[bone] = RPM_REST_ROTATIONS_XYZW[bone]

    logger.info(f"Extracted rest quaternions for {len(result)} bones from custom GLB")
    return result


# ── Rest pose helpers (parameterized) ──────────────────────────────

# Module-level caches for the default RPM rest pose (avoids recomputation)
_rpm_rest_matrices: dict[str, np.ndarray] = {}
_rpm_world_rest_matrices: dict[str, np.ndarray] = {}


def _get_rest_matrix(bone_name: str, rest_rotations: dict) -> np.ndarray:
    """Get 3x3 rest rotation matrix for a bone.

    Uses module-level cache only when rest_rotations is the default RPM dict.
    """
    # Fast path: use cache for default RPM rest rotations
    if rest_rotations is RPM_REST_ROTATIONS_XYZW:
        if bone_name not in _rpm_rest_matrices:
            if bone_name in rest_rotations:
                xyzw = rest_rotations[bone_name]
                _rpm_rest_matrices[bone_name] = Rotation.from_quat(xyzw).as_matrix()
            else:
                _rpm_rest_matrices[bone_name] = np.eye(3)
        return _rpm_rest_matrices[bone_name]

    # Custom rest rotations: compute directly (no caching)
    if bone_name in rest_rotations:
        return Rotation.from_quat(rest_rotations[bone_name]).as_matrix()
    return np.eye(3)


def _get_world_rest_matrix(bone_name: str, rest_rotations: dict) -> np.ndarray:
    """Get accumulated world rest rotation for a bone.

    W_bone = W_parent × R_rest_local
    """
    # Fast path: use cache for default RPM rest rotations
    if rest_rotations is RPM_REST_ROTATIONS_XYZW:
        if bone_name not in _rpm_world_rest_matrices:
            local_rest = _get_rest_matrix(bone_name, rest_rotations)
            parent = MIXAMO_PARENT.get(bone_name)
            if parent is None:
                _rpm_world_rest_matrices[bone_name] = local_rest.copy()
            else:
                parent_world = _get_world_rest_matrix(parent, rest_rotations)
                _rpm_world_rest_matrices[bone_name] = parent_world @ local_rest
        return _rpm_world_rest_matrices[bone_name]

    # Custom rest rotations: compute directly
    local_rest = _get_rest_matrix(bone_name, rest_rotations)
    parent = MIXAMO_PARENT.get(bone_name)
    if parent is None:
        return local_rest.copy()
    parent_world = _get_world_rest_matrix(parent, rest_rotations)
    return parent_world @ local_rest


def _retarget_rotation(smpl_rot: np.ndarray, mixamo_name: str,
                       rest_rotations: dict) -> np.ndarray:
    """Apply the standard retargeting formula for one bone across all frames.

    Formula (simplified for SMPL source where all bind rotations = identity):
        trgLocal = inv(W_trg_parent) × R_smpl × W_trg_parent × R_rest_local

    Where W_trg_parent is the accumulated world rest rotation of the target
    bone's parent in the target skeleton.

    Args:
        smpl_rot: (N, 3, 3) SMPL local rotation matrices for this joint.
        mixamo_name: Target Mixamo bone name.
        rest_rotations: Dict mapping bone name → (x,y,z,w) rest quaternion.

    Returns:
        (N, 3, 3) retargeted rotation matrices.
    """
    rest_local = _get_rest_matrix(mixamo_name, rest_rotations)
    parent = MIXAMO_PARENT.get(mixamo_name)

    if parent is None:
        # Root bone: parent world = identity, formula simplifies to R_smpl × R_rest
        return smpl_rot @ rest_local

    parent_world = _get_world_rest_matrix(parent, rest_rotations)
    inv_parent_world = parent_world.T  # inverse of rotation matrix = transpose

    # trgLocal = inv(Wp) @ R_smpl @ Wp @ R_rest_local
    # Shapes: (3,3) @ (N,3,3) @ (3,3) → (N,3,3)  (numpy broadcasts)
    combined_right = parent_world @ rest_local  # (3, 3)
    return inv_parent_world @ smpl_rot @ combined_right


# Backward-compatible aliases for tests that import the old names
def _get_rpm_rest_matrix(bone_name: str) -> np.ndarray:
    return _get_rest_matrix(bone_name, RPM_REST_ROTATIONS_XYZW)

def _get_rpm_world_rest_matrix(bone_name: str) -> np.ndarray:
    return _get_world_rest_matrix(bone_name, RPM_REST_ROTATIONS_XYZW)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D continuous rotation representation to 3x3 rotation matrix.

    Based on "On the Continuity of Rotation Representations in Neural Networks"
    (Zhou et al., 2019). HY-Motion stores the 6D values as a 3x2 matrix in
    row-major order: [r00, r01, r10, r11, r20, r21], where column 0 and
    column 1 are the two basis vectors to orthogonalize.

    Args:
        rot6d: Array of shape (..., 6) — interleaved columns of a 3x2 matrix.

    Returns:
        Array of shape (..., 3, 3) rotation matrices.
    """
    shape = rot6d.shape[:-1]
    # Reshape to (..., 3, 2) and extract columns — matches HY-Motion's format
    x = rot6d.reshape(*shape, 3, 2)
    a1 = x[..., 0]  # first column vector
    a2 = x[..., 1]  # second column vector

    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1).reshape(*shape, 3, 3)


def _matrix_to_euler(matrices: np.ndarray, order: str = "ZXY") -> np.ndarray:
    """Convert rotation matrices to Euler angles (degrees).

    Args:
        matrices: Array of shape (..., 3, 3).
        order: Euler angle order (e.g. "ZXY" for BVH).

    Returns:
        Array of shape (..., 3) in degrees.
    """
    orig_shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    r = Rotation.from_matrix(flat)
    # scipy uses lowercase intrinsic convention
    euler = r.as_euler(order.lower(), degrees=True)
    return euler.reshape(*orig_shape, 3)


def _matrix_to_quaternion(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to quaternions (w, x, y, z).

    Args:
        matrices: Array of shape (..., 3, 3).

    Returns:
        Array of shape (..., 4) as (w, x, y, z).
    """
    orig_shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    r = Rotation.from_matrix(flat)
    # scipy returns (x, y, z, w), we want (w, x, y, z)
    quat_xyzw = r.as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    return quat_wxyz.reshape(*orig_shape, 4)


def _ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """Ensure quaternion sequence is continuous for smooth SLERP interpolation.

    Handles two issues:
    1. Sign flips: q and -q represent the same rotation. Fixed by negating
       when dot(q[i], q[i-1]) < 0.
    2. 180° gimbal lock: at rotation angle ≈ 180°, the axis is ill-determined
       by matrix→quaternion conversion (scipy picks arbitrarily). When |w| ≈ 0
       and the frame-to-frame delta exceeds a threshold, replace the erratic
       quaternion with a SLERP interpolation from stable neighbors.

    Args:
        quats: (N, 4) array of quaternions in (w, x, y, z) order.

    Returns:
        (N, 4) array with continuous quaternions.
    """
    quats = quats.copy()
    n = len(quats)
    if n < 3:
        return quats

    # Pass 0: normalize frame 0 to positive-w hemisphere so exported
    # quaternions match the rest rotation sign convention (w ≥ 0).
    if quats[0][0] < 0:
        quats[0] = -quats[0]

    # Pass 1: basic sign continuity
    for i in range(1, n):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    # Pass 2: fix 180° axis instability
    # Detect frames near 180° with large deltas and interpolate them
    ANGLE_THRESHOLD = 165  # degrees — rotation angle above which axis is unstable
    DELTA_THRESHOLD = 15   # degrees — max expected per-frame delta at 30fps
    W_THRESHOLD = np.cos(np.radians(ANGLE_THRESHOLD) / 2)  # |w| below this = near 180°

    # Find unstable frames
    unstable = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if abs(quats[i][0]) < W_THRESHOLD:  # w component near 0 = angle near 180°
            dot = np.clip(np.dot(quats[i], quats[i - 1]), -1, 1)
            delta_deg = np.degrees(2 * np.arccos(abs(dot)))
            if delta_deg > DELTA_THRESHOLD:
                unstable[i] = True

    if not np.any(unstable):
        return quats

    # For each run of unstable frames, SLERP from the last stable frame
    # before the run to the first stable frame after the run
    i = 0
    while i < n:
        if not unstable[i]:
            i += 1
            continue

        # Find the run of unstable frames
        start = i
        while i < n and unstable[i]:
            i += 1
        end = i  # exclusive

        # Find stable anchors
        anchor_before = start - 1 if start > 0 else 0
        anchor_after = end if end < n else n - 1

        # SLERP between anchors
        r_before = Rotation.from_quat([quats[anchor_before][1], quats[anchor_before][2],
                                       quats[anchor_before][3], quats[anchor_before][0]])
        r_after = Rotation.from_quat([quats[anchor_after][1], quats[anchor_after][2],
                                      quats[anchor_after][3], quats[anchor_after][0]])

        total_span = anchor_after - anchor_before
        if total_span == 0:
            continue

        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], Rotation.concatenate([r_before, r_after]))

        for j in range(start, end):
            t = (j - anchor_before) / total_span
            r_interp = slerp(t)
            q_xyzw = r_interp.as_quat()
            quats[j] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        logger.info(f"Smoothed {end - start} unstable frames ({start}-{end-1}) near 180° rotation")

    # Pass 3: re-check sign continuity after interpolation
    for i in range(1, n):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    return quats


def make_loopable(retargeted: dict, blend_frames: int = 15) -> dict:
    """Post-process a retargeted animation to loop seamlessly via crossfade.

    Blends the last `blend_frames` of the animation toward frame 0 using
    SLERP (for rotations) and linear interpolation (for root translation).
    The weight spacing is uniform so the wrap from the last frame back to
    frame 0 has the same delta as between any two adjacent blend frames.

    Args:
        retargeted: Output from retarget_to_mixamo().
        blend_frames: Number of frames at the end to crossfade.
            Capped at 1/3 of total frames to avoid over-blending.

    Returns:
        The same dict, modified in place.
    """
    from scipy.spatial.transform import Slerp as ScipySlerp

    n = retargeted["num_frames"]
    blend = min(blend_frames, n // 3)
    if blend < 2:
        logger.warning(f"Animation too short ({n} frames) to loop with {blend_frames}-frame blend")
        return retargeted

    start = n - blend  # first frame in the blend region

    for name, data in retargeted["bones"].items():
        rots = data["rotations"]  # (N, 3, 3) numpy array

        # SLERP each blend frame toward frame 0
        q_target = Rotation.from_matrix(rots[0])
        for i in range(start, n):
            t = (i - start + 1) / (blend + 1)  # uniform: 1/(B+1) … B/(B+1)
            q_current = Rotation.from_matrix(rots[i])
            slerp = ScipySlerp([0, 1], Rotation.concatenate([q_current, q_target]))
            rots[i] = slerp(t).as_matrix()

        # Recompute Euler angles from the blended matrices
        data["euler_zxy"] = _matrix_to_euler(rots, "ZXY")

        # Lerp root translation toward frame 0
        if data["position"] is not None:
            pos = data["position"]  # (N, 3)
            p0 = pos[0].copy()
            for i in range(start, n):
                t = (i - start + 1) / (blend + 1)
                pos[i] = (1 - t) * pos[i] + t * p0

    logger.info(f"Applied loop crossfade: {blend} blend frames out of {n} total")
    return retargeted


def retarget_to_mixamo(
    rot6d: np.ndarray,
    transl: np.ndarray,
    fps: int = 30,
    zero_root_xz: bool = False,
    scale: float = 1.0,
    rest_rotations: dict[str, tuple] | None = None,
) -> dict:
    """Retarget SMPL-H motion data to a Mixamo-compatible skeleton.

    Args:
        rot6d: Joint rotations in 6D format, shape (num_frames, 22, 6).
            Joint 0 is root, joints 1-21 are body joints in SMPL order.
        transl: Root translation, shape (num_frames, 3).
        fps: Frame rate.
        zero_root_xz: If True, zero out root XZ translation (for "in place" anims).
        scale: Scale factor for positions (1.0 = meters, 100.0 = centimeters for FBX).
        rest_rotations: Optional custom rest pose quaternions {bone: (x,y,z,w)}.
            If None, uses the default RPM rest pose.

    Returns:
        dict mapping Mixamo bone names to rotation data:
        {
            "fps": int,
            "num_frames": int,
            "bones": {
                "Hips": {
                    "rotations": np.ndarray (num_frames, 3, 3),
                    "euler_zxy": np.ndarray (num_frames, 3),  # degrees
                    "position": np.ndarray (num_frames, 3) or None,
                },
                ...
            }
        }
    """
    if rest_rotations is None:
        rest_rotations = RPM_REST_ROTATIONS_XYZW

    n_frames = rot6d.shape[0]

    # Convert all 6D rotations to matrices: (N, 22, 6) → (N, 22, 3, 3)
    all_rot_matrices = _rot6d_to_matrix(rot6d)

    bones = {}

    # Root bone (Hips) — gets both rotation and translation
    root_trans = transl.copy()
    if zero_root_xz:
        root_trans[:, [0, 2]] = 0.0  # zero X and Z, keep Y (vertical)
    root_trans *= scale

    root_rot = _retarget_rotation(all_rot_matrices[:, 0], "Hips", rest_rotations)

    bones["Hips"] = {
        "rotations": root_rot,
        "euler_zxy": _matrix_to_euler(root_rot, "ZXY"),
        "position": root_trans,
    }

    # Map each SMPL joint to its Mixamo equivalent
    for smpl_idx, mixamo_name in SMPL_TO_MIXAMO.items():
        if smpl_idx == 0:
            continue  # Already handled root
        if smpl_idx in SMPL_SKIP_JOINTS:
            continue

        # rot6d joint indices: 0=root, 1-21=body joints matching SMPL joints 1-21
        if smpl_idx >= all_rot_matrices.shape[1]:
            logger.warning(f"Joint index {smpl_idx} ({mixamo_name}) out of range, skipping")
            continue

        rot = _retarget_rotation(all_rot_matrices[:, smpl_idx], mixamo_name, rest_rotations)

        bones[mixamo_name] = {
            "rotations": rot,
            "euler_zxy": _matrix_to_euler(rot, "ZXY"),
            "position": None,
        }

    logger.info(f"Retargeted {len(bones)} bones across {n_frames} frames")

    return {
        "fps": fps,
        "num_frames": n_frames,
        "bones": bones,
    }


def export_bvh(retargeted: dict, output_path: str) -> str:
    """Export retargeted animation as BVH file.

    BVH is a simple text format supported by Blender, Godot, Three.js, etc.

    Args:
        retargeted: Output from retarget_to_mixamo().
        output_path: File path to write.

    Returns:
        The output path.
    """
    fps = retargeted["fps"]
    n_frames = retargeted["num_frames"]
    bones = retargeted["bones"]
    frame_time = 1.0 / fps

    lines = []

    # ── HIERARCHY section ────────────────────────────────────────
    lines.append("HIERARCHY")

    def write_joint(name: str, depth: int, is_root: bool = False):
        indent = "  " * depth
        if is_root:
            lines.append(f"{indent}ROOT {name}")
        else:
            lines.append(f"{indent}JOINT {name}")

        lines.append(f"{indent}{{")

        # Offset — use zero for now (skeleton proportions come from the avatar)
        lines.append(f"{indent}  OFFSET 0.0 0.0 0.0")

        if is_root:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

        children = MIXAMO_HIERARCHY.get(name, [])
        if children:
            for child in children:
                write_joint(child, depth + 1)
        else:
            # End site for leaf nodes
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.0 1.0 0.0")
            lines.append(f"{indent}  }}")

        lines.append(f"{indent}}}")

    write_joint("Hips", 0, is_root=True)

    # ── MOTION section ───────────────────────────────────────────
    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    def get_bone_values(name: str, frame: int) -> list:
        """Get the channel values for a bone at a given frame."""
        values = []
        bone = bones.get(name)

        if name == "Hips" and bone is not None:
            # Root has position + rotation
            pos = bone["position"][frame]
            values.extend([pos[0], pos[1], pos[2]])

        if bone is not None:
            euler = bone["euler_zxy"][frame]
            values.extend([euler[0], euler[1], euler[2]])
        else:
            # Bone not in SMPL output — use identity rotation
            values.extend([0.0, 0.0, 0.0])

        return values

    # Write frame data in hierarchy traversal order
    def collect_bone_order(name: str, order: list):
        order.append(name)
        for child in MIXAMO_HIERARCHY.get(name, []):
            collect_bone_order(child, order)

    bone_order = []
    collect_bone_order("Hips", bone_order)

    for frame in range(n_frames):
        frame_values = []
        for bone_name in bone_order:
            frame_values.extend(get_bone_values(bone_name, frame))
        lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    bvh_content = "\n".join(lines) + "\n"

    with open(output_path, "w") as f:
        f.write(bvh_content)

    logger.info(f"Exported BVH to {output_path} ({n_frames} frames, {len(bone_order)} bones)")
    return output_path


def export_glb_animation(retargeted: dict, output_path: str) -> str:
    """Export retargeted animation as a glTF binary (.glb) animation clip.

    This creates a minimal GLB containing only animation data (no mesh),
    which can be loaded by Three.js or Godot and applied to an RPM avatar.

    Args:
        retargeted: Output from retarget_to_mixamo().
        output_path: File path to write.

    Returns:
        The output path.
    """
    return _export_gltf(retargeted, output_path)


def export_gltf_animation(retargeted: dict, output_path: str) -> str:
    """Export retargeted animation as ASCII glTF (.gltf) for debugging.

    Same data as GLB but in human-readable JSON with base64-encoded buffers.

    Args:
        retargeted: Output from retarget_to_mixamo().
        output_path: File path to write (.gltf).

    Returns:
        The output path.
    """
    return _export_gltf(retargeted, output_path)


def _export_gltf(retargeted: dict, output_path: str) -> str:
    """Internal: export retargeted animation as glTF (.gltf) or GLB (.glb).

    Format is determined by file extension. .gltf uses embedded base64 data URIs
    for easy inspection.

    Args:
        retargeted: Output from retarget_to_mixamo().
        output_path: File path to write.

    Returns:
        The output path.
    """
    try:
        import pygltflib
    except ImportError:
        logger.error("pygltflib not installed — cannot export. Install with: pip install pygltflib")
        raise

    fps = retargeted["fps"]
    n_frames = retargeted["num_frames"]
    bones = retargeted["bones"]

    # Build glTF animation with quaternion tracks for each bone
    timestamps = np.linspace(0, n_frames / fps, n_frames, dtype=np.float32)

    nodes = []
    animations_channels = []
    animations_samplers = []
    accessors = []
    buffer_views = []
    buffer_data = bytearray()

    def add_buffer(data: np.ndarray) -> int:
        """Add data to the buffer and return the accessor index."""
        data = np.ascontiguousarray(data)
        raw = data.tobytes()
        offset = len(buffer_data)
        buffer_data.extend(raw)
        # Pad to 4-byte alignment
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)

        bv_idx = len(buffer_views)
        buffer_views.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(raw),
        ))

        acc_idx = len(accessors)
        if data.ndim == 1:
            component_type = pygltflib.FLOAT
            acc_type = pygltflib.SCALAR
            count = len(data)
            mins = [float(data.min())]
            maxs = [float(data.max())]
        elif data.shape[1] == 3:
            component_type = pygltflib.FLOAT
            acc_type = pygltflib.VEC3
            count = data.shape[0]
            mins = data.min(axis=0).tolist()
            maxs = data.max(axis=0).tolist()
        elif data.shape[1] == 4:
            component_type = pygltflib.FLOAT
            acc_type = pygltflib.VEC4
            count = data.shape[0]
            mins = data.min(axis=0).tolist()
            maxs = data.max(axis=0).tolist()
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        accessors.append(pygltflib.Accessor(
            bufferView=bv_idx,
            componentType=component_type,
            count=count,
            type=acc_type,
            max=maxs,
            min=mins,
        ))

        return acc_idx

    # Create a node for each bone
    bone_order = []

    def collect_bone_order(name, order):
        order.append(name)
        for child in MIXAMO_HIERARCHY.get(name, []):
            collect_bone_order(child, order)

    collect_bone_order("Hips", bone_order)

    node_indices = {}
    for bone_name in bone_order:
        node_idx = len(nodes)
        node_indices[bone_name] = node_idx
        children_indices = [
            node_indices.get(c) for c in MIXAMO_HIERARCHY.get(bone_name, [])
            if c in node_indices
        ]
        nodes.append(pygltflib.Node(
            name=bone_name,
            children=[i for i in children_indices if i is not None],
        ))

    # Fix children references (they may not exist yet during forward pass)
    for bone_name in bone_order:
        node_idx = node_indices[bone_name]
        children = MIXAMO_HIERARCHY.get(bone_name, [])
        nodes[node_idx].children = [node_indices[c] for c in children if c in node_indices]

    # Add timestamp accessor (shared across all channels)
    time_acc_idx = add_buffer(timestamps)

    # Add animation channels for each bone
    for bone_name in bone_order:
        node_idx = node_indices[bone_name]
        bone = bones.get(bone_name)

        if bone is not None:
            # Rotation channel (quaternions)
            quats = _matrix_to_quaternion(bone["rotations"])  # (N, 4) as (w,x,y,z)
            quats = _ensure_quaternion_continuity(quats)
            # glTF expects (x, y, z, w)
            quats_xyzw = np.concatenate([quats[:, 1:4], quats[:, 0:1]], axis=-1).astype(np.float32)
            rot_acc_idx = add_buffer(quats_xyzw)

            sampler_idx = len(animations_samplers)
            animations_samplers.append(pygltflib.AnimationSampler(
                input=time_acc_idx,
                output=rot_acc_idx,
                interpolation=pygltflib.LINEAR,
            ))
            animations_channels.append(pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(
                    node=node_idx,
                    path="rotation",
                ),
            ))

            # Position channel (only for root)
            if bone["position"] is not None:
                pos = bone["position"].astype(np.float32)
                pos_acc_idx = add_buffer(pos)

                sampler_idx = len(animations_samplers)
                animations_samplers.append(pygltflib.AnimationSampler(
                    input=time_acc_idx,
                    output=pos_acc_idx,
                    interpolation=pygltflib.LINEAR,
                ))
                animations_channels.append(pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_idx,
                        path="translation",
                    ),
                ))

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[node_indices["Hips"]])],
        nodes=nodes,
        animations=[pygltflib.Animation(
            name="generated",
            channels=animations_channels,
            samplers=animations_samplers,
        )],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=len(buffer_data))],
    )

    gltf.set_binary_blob(bytes(buffer_data))

    is_ascii = output_path.lower().endswith(".gltf")
    if is_ascii:
        # Convert binary buffer to base64 data URI for single-file readable output
        gltf.convert_buffers(pygltflib.BufferFormat.DATAURI)

    gltf.save(output_path)

    fmt_label = "glTF (ASCII)" if is_ascii else "GLB"
    logger.info(f"Exported {fmt_label} animation to {output_path} ({n_frames} frames, {len(bone_order)} bones)")
    return output_path
