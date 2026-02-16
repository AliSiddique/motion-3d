"""Compare RPM default rest poses vs extracted rest poses from a real avatar GLB.

Prints side-by-side comparison of quaternions and the resulting retargeted
rotations for the same synthetic input, to diagnose retargeting differences.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retargeter import (
    RPM_REST_ROTATIONS_XYZW,
    SMPL_TO_MIXAMO,
    _get_rest_matrix,
    _get_world_rest_matrix,
    _matrix_to_quaternion,
    _retarget_rotation,
    _rot6d_to_matrix,
    extract_rest_from_glb,
    retarget_to_mixamo,
    export_gltf_animation,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_avatar.glb")

with open(FIXTURE, "rb") as f:
    glb_bytes = f.read()

extracted = extract_rest_from_glb(glb_bytes)

print("=" * 90)
print(f"{'Bone':<20} {'RPM default (x,y,z,w)':<35} {'Extracted (x,y,z,w)':<35} {'dot':>6}")
print("=" * 90)

for name in sorted(RPM_REST_ROTATIONS_XYZW.keys()):
    rpm = np.array(RPM_REST_ROTATIONS_XYZW[name])
    ext = np.array(extracted.get(name, (0, 0, 0, 1)))
    dot = abs(np.dot(rpm, ext))
    marker = " <<<" if dot < 0.95 else ""
    print(f"{name:<20} ({rpm[0]:+7.4f},{rpm[1]:+7.4f},{rpm[2]:+7.4f},{rpm[3]:+7.4f})   "
          f"({ext[0]:+7.4f},{ext[1]:+7.4f},{ext[2]:+7.4f},{ext[3]:+7.4f})   {dot:.4f}{marker}")

# Check what bones are in the GLB but NOT in our defaults
extra = set(extracted.keys()) - set(RPM_REST_ROTATIONS_XYZW.keys())
if extra:
    print(f"\nExtra bones in GLB (not in defaults): {extra}")

missing = set(RPM_REST_ROTATIONS_XYZW.keys()) - set(extracted.keys())
if missing:
    print(f"\nMissing bones in GLB (not extracted): {missing}")

# Now compare retargeted output for a simple identity pose
print("\n\n" + "=" * 90)
print("Retargeted frame 0 (identity SMPL input) — RPM vs Extracted")
print("=" * 90)

# Identity rot6d
rot6d = np.tile(np.array([1, 0, 0, 1, 0, 0], dtype=np.float64), (5, 22, 1))
transl = np.zeros((5, 3))
transl[:, 1] = 0.9

rpm_result = retarget_to_mixamo(rot6d, transl, fps=30, rest_rotations=RPM_REST_ROTATIONS_XYZW)
ext_result = retarget_to_mixamo(rot6d, transl, fps=30, rest_rotations=extracted)

print(f"\n{'Bone':<20} {'RPM quat (x,y,z,w)':<35} {'Ext quat (x,y,z,w)':<35} {'delta°':>8}")
print("-" * 90)

for name in sorted(rpm_result["bones"].keys()):
    rpm_rot = rpm_result["bones"][name]["rotations"][0]  # (3,3)
    ext_rot = ext_result["bones"][name]["rotations"][0]  # (3,3)

    rpm_q = Rotation.from_matrix(rpm_rot).as_quat()  # (x,y,z,w)
    ext_q = Rotation.from_matrix(ext_rot).as_quat()  # (x,y,z,w)

    # Angular difference
    r_diff = Rotation.from_matrix(ext_rot) * Rotation.from_matrix(rpm_rot).inv()
    delta_deg = np.degrees(r_diff.magnitude())

    marker = " <<<" if delta_deg > 5 else ""
    print(f"{name:<20} ({rpm_q[0]:+7.4f},{rpm_q[1]:+7.4f},{rpm_q[2]:+7.4f},{rpm_q[3]:+7.4f})   "
          f"({ext_q[0]:+7.4f},{ext_q[1]:+7.4f},{ext_q[2]:+7.4f},{ext_q[3]:+7.4f})   {delta_deg:7.2f}°{marker}")

# Also dump the extracted GLB JSON to see what bone names actually exist
print("\n\n" + "=" * 90)
print("Raw GLB node names and rotations")
print("=" * 90)

import struct
magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
json_len, json_type = struct.unpack_from("<II", glb_bytes, 12)
gltf = json.loads(glb_bytes[20:20 + json_len])

for node in gltf.get("nodes", []):
    name = node.get("name", "")
    rot = node.get("rotation")
    if rot:
        print(f"  {name:<40} rotation: [{rot[0]:+.6f}, {rot[1]:+.6f}, {rot[2]:+.6f}, {rot[3]:+.6f}]")
    elif "Arm" in name or "Leg" in name or "Spine" in name or "Hips" in name or "Head" in name or "Foot" in name or "Hand" in name or "Neck" in name or "Shoulder" in name:
        print(f"  {name:<40} rotation: NONE (identity)")
