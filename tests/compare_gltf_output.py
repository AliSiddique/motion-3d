"""Export the same animation with RPM defaults vs extracted rest, and compare the glTF JSON.

Focuses on the quaternion values for LeftArm (biggest difference) and Hips (root).
"""
import json
import os
import sys
import base64
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retargeter import (
    RPM_REST_ROTATIONS_XYZW,
    extract_rest_from_glb,
    retarget_to_mixamo,
    export_gltf_animation,
)

# Load real avatar
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_avatar.glb")
with open(FIXTURE, "rb") as f:
    glb_bytes = f.read()

extracted = extract_rest_from_glb(glb_bytes)

# Create a simple wave motion (same as tests)
def _rot6d_from_scipy(r):
    mat = r.as_matrix()
    return np.array([mat[0,0], mat[0,1], mat[1,0], mat[1,1], mat[2,0], mat[2,1]])

n_frames = 30
rot6d = np.tile(np.array([1,0,0,1,0,0], dtype=np.float64), (n_frames, 22, 1))
transl = np.zeros((n_frames, 3))
transl[:, 1] = 0.9

for i in range(n_frames):
    t = i / n_frames
    angle = np.sin(t * 2 * np.pi) * np.radians(45)
    r_arm = Rotation.from_euler("z", angle)
    rot6d[i, 17] = _rot6d_from_scipy(r_arm)  # RightArm

# Retarget with both rest poses
rpm_result = retarget_to_mixamo(rot6d.copy(), transl.copy(), fps=30, rest_rotations=RPM_REST_ROTATIONS_XYZW)
ext_result = retarget_to_mixamo(rot6d.copy(), transl.copy(), fps=30, rest_rotations=extracted)

# Export both to glTF ASCII
with tempfile.TemporaryDirectory() as tmp:
    rpm_path = os.path.join(tmp, "rpm.gltf")
    ext_path = os.path.join(tmp, "ext.gltf")
    export_gltf_animation(rpm_result, rpm_path)
    export_gltf_animation(ext_result, ext_path)

    with open(rpm_path) as f:
        rpm_gltf = json.load(f)
    with open(ext_path) as f:
        ext_gltf = json.load(f)

# Parse buffers
rpm_buf = base64.b64decode(rpm_gltf["buffers"][0]["uri"].split(",")[1])
ext_buf = base64.b64decode(ext_gltf["buffers"][0]["uri"].split(",")[1])

# Build node name → index mapping
rpm_nodes = {n["name"]: i for i, n in enumerate(rpm_gltf["nodes"])}
ext_nodes = {n["name"]: i for i, n in enumerate(ext_gltf["nodes"])}

# For each bone, find the rotation channel and extract quaternion data
def get_bone_quats(gltf, buf, bone_name):
    node_idx = None
    for i, n in enumerate(gltf["nodes"]):
        if n["name"] == bone_name:
            node_idx = i
            break
    if node_idx is None:
        return None

    for ch in gltf["animations"][0]["channels"]:
        if ch["target"]["node"] == node_idx and ch["target"]["path"] == "rotation":
            sampler = gltf["animations"][0]["samplers"][ch["sampler"]]
            acc = gltf["accessors"][sampler["output"]]
            bv = gltf["bufferViews"][acc["bufferView"]]
            data = np.frombuffer(buf, dtype=np.float32,
                                  count=acc["count"] * 4,
                                  offset=bv["byteOffset"]).reshape(-1, 4)
            return data  # (x, y, z, w) glTF order
    return None

# Compare key bones
bones_to_check = ["Hips", "LeftArm", "RightArm", "LeftForeArm", "LeftShoulder", "Spine", "LeftHand"]

for bone in bones_to_check:
    rpm_quats = get_bone_quats(rpm_gltf, rpm_buf, bone)
    ext_quats = get_bone_quats(ext_gltf, ext_buf, bone)

    if rpm_quats is None or ext_quats is None:
        print(f"{bone}: MISSING")
        continue

    # Also get the avatar's rest quaternion
    avatar_rest = extracted.get(bone, (0, 0, 0, 1))
    rpm_rest = RPM_REST_ROTATIONS_XYZW.get(bone, (0, 0, 0, 1))

    print(f"\n{'=' * 80}")
    print(f"BONE: {bone}")
    print(f"  Avatar GLB rest:  ({avatar_rest[0]:+.4f}, {avatar_rest[1]:+.4f}, {avatar_rest[2]:+.4f}, {avatar_rest[3]:+.4f})")
    print(f"  RPM default rest: ({rpm_rest[0]:+.4f}, {rpm_rest[1]:+.4f}, {rpm_rest[2]:+.4f}, {rpm_rest[3]:+.4f})")
    print(f"  {'Frame':<8} {'RPM animation quat (xyzw)':<40} {'Extracted animation quat (xyzw)':<40} {'delta°'}")
    print(f"  {'-' * 96}")

    for frame in [0, 7, 14, 21, 29]:
        rq = rpm_quats[frame]
        eq = ext_quats[frame]

        # Angular difference
        r_rpm = Rotation.from_quat(rq)
        r_ext = Rotation.from_quat(eq)
        delta = np.degrees((r_ext * r_rpm.inv()).magnitude())

        print(f"  {frame:<8} ({rq[0]:+.4f},{rq[1]:+.4f},{rq[2]:+.4f},{rq[3]:+.4f})"
              f"   ({eq[0]:+.4f},{eq[1]:+.4f},{eq[2]:+.4f},{eq[3]:+.4f})"
              f"   {delta:7.2f}°")

    # Check: at frame 0 (identity SMPL), should the animation quat match the rest?
    print(f"\n  Frame 0 vs avatar GLB rest:")
    r_ext0 = Rotation.from_quat(ext_quats[0])
    r_rest = Rotation.from_quat(np.array(avatar_rest))
    match_delta = np.degrees((r_ext0 * r_rest.inv()).magnitude())
    print(f"    Extracted anim frame 0 vs avatar rest: {match_delta:.4f}° (should be 0)")

    r_rpm0 = Rotation.from_quat(rpm_quats[0])
    rpm_match = np.degrees((r_rpm0 * r_rest.inv()).magnitude())
    print(f"    RPM anim frame 0 vs avatar rest:       {rpm_match:.4f}° (non-zero = mismatch)")

# Also dump the node hierarchy of both
print(f"\n\n{'=' * 80}")
print("Node hierarchy comparison")
print("=" * 80)
print(f"\nAnimation glTF nodes:")
for i, n in enumerate(rpm_gltf["nodes"]):
    children = n.get("children", [])
    child_names = [rpm_gltf["nodes"][c]["name"] for c in children]
    print(f"  [{i}] {n['name']} → children: {child_names}")

print(f"\nAvatar GLB bone nodes (from raw JSON):")
import struct
magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
json_len, json_type = struct.unpack_from("<II", glb_bytes, 12)
gltf = json.loads(glb_bytes[20:20 + json_len])
for i, n in enumerate(gltf.get("nodes", [])):
    name = n.get("name", "")
    if any(b in name for b in ["Hips", "Spine", "Arm", "Leg", "Foot", "Hand",
                                  "Head", "Neck", "Shoulder", "Armature"]):
        children = n.get("children", [])
        child_names = [gltf["nodes"][c]["name"] for c in children]
        print(f"  [{i}] {name} → children: {child_names}")
