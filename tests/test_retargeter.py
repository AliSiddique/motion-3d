"""Unit tests for SMPL → Mixamo/RPM retargeting.

Tests quaternion conversion, retargeting formula, and GLB/glTF export integrity
using synthetic SMPL rotation data (no model files needed).
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.retargeter import (
    MIXAMO_HIERARCHY,
    MIXAMO_PARENT,
    RPM_REST_ROTATIONS_XYZW,
    SMPL_TO_MIXAMO,
    _get_rpm_rest_matrix,
    _get_rpm_world_rest_matrix,
    _matrix_to_euler,
    _matrix_to_quaternion,
    _retarget_rotation,
    _rot6d_to_matrix,
    export_bvh,
    export_glb_animation,
    export_gltf_animation,
    extract_rest_from_glb,
    make_loopable,
    retarget_to_mixamo,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _identity_rot6d() -> np.ndarray:
    """Identity rotation in 6D format: columns of I₃ → [1,0, 0,1, 0,0]."""
    return np.array([1, 0, 0, 1, 0, 0], dtype=np.float64)


def _rot6d_from_scipy(r: Rotation) -> np.ndarray:
    """Convert a scipy Rotation to 6D representation (HY-Motion format)."""
    mat = r.as_matrix()  # (3, 3)
    # 6D = first two columns interleaved as [r00,r01, r10,r11, r20,r21]
    return np.array([
        mat[0, 0], mat[0, 1],
        mat[1, 0], mat[1, 1],
        mat[2, 0], mat[2, 1],
    ])


def _make_smpl_frames(n_frames: int, n_joints: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """Create identity SMPL data: all joints at identity rotation, root at origin."""
    rot6d = np.tile(_identity_rot6d(), (n_frames, n_joints, 1))
    transl = np.zeros((n_frames, 3))
    transl[:, 1] = 0.9  # hip height ~0.9m
    return rot6d, transl


def _make_wave_animation(n_frames: int = 30, fps: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 'wave' animation.

    Right arm (SMPL joint 17) oscillates ±45° around Z axis.
    Right forearm (SMPL joint 19) oscillates ±30° around Z axis with phase offset.
    Everything else stays at identity.
    """
    rot6d = np.tile(_identity_rot6d(), (n_frames, 22, 1))
    transl = np.zeros((n_frames, 3))
    transl[:, 1] = 0.9

    for i in range(n_frames):
        t = i / n_frames
        # Right shoulder (joint 17 → RightArm): wave up/down
        angle_arm = np.sin(t * 2 * np.pi) * np.radians(45)
        r_arm = Rotation.from_euler("z", angle_arm)
        rot6d[i, 17] = _rot6d_from_scipy(r_arm)

        # Right forearm (joint 19 → RightForeArm): secondary wave
        angle_forearm = np.sin(t * 2 * np.pi + np.pi / 4) * np.radians(30)
        r_forearm = Rotation.from_euler("z", angle_forearm)
        rot6d[i, 19] = _rot6d_from_scipy(r_forearm)

    return rot6d, transl


# ── rot6d → matrix ──────────────────────────────────────────────────


class TestRot6dToMatrix:
    def test_identity(self):
        rot6d = _identity_rot6d().reshape(1, 6)
        mat = _rot6d_to_matrix(rot6d)
        np.testing.assert_allclose(mat[0], np.eye(3), atol=1e-7)

    def test_90deg_z(self):
        """90° around Z: columns = [0,-1,0], [1,0,0], [0,0,1]."""
        r = Rotation.from_euler("z", 90, degrees=True)
        rot6d = _rot6d_from_scipy(r).reshape(1, 6)
        mat = _rot6d_to_matrix(rot6d)
        expected = r.as_matrix()
        np.testing.assert_allclose(mat[0], expected, atol=1e-6)

    def test_arbitrary_rotation(self):
        r = Rotation.from_euler("xyz", [30, 45, 60], degrees=True)
        rot6d = _rot6d_from_scipy(r).reshape(1, 6)
        mat = _rot6d_to_matrix(rot6d)
        expected = r.as_matrix()
        np.testing.assert_allclose(mat[0], expected, atol=1e-6)

    def test_batch(self):
        rotations = Rotation.random(10, random_state=42)
        rot6d = np.array([_rot6d_from_scipy(r) for r in rotations])
        mats = _rot6d_to_matrix(rot6d)
        for i, r in enumerate(rotations):
            np.testing.assert_allclose(mats[i], r.as_matrix(), atol=1e-6)

    def test_orthogonality(self):
        """Output matrices must be proper rotations (det=1, R^T R = I)."""
        rotations = Rotation.random(20, random_state=123)
        rot6d = np.array([_rot6d_from_scipy(r) for r in rotations])
        mats = _rot6d_to_matrix(rot6d)
        for m in mats:
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-6)


# ── matrix → quaternion ─────────────────────────────────────────────


class TestMatrixToQuaternion:
    def test_identity(self):
        mat = np.eye(3).reshape(1, 3, 3)
        quat = _matrix_to_quaternion(mat)  # (w, x, y, z)
        # Identity quaternion: (1, 0, 0, 0) or (-1, 0, 0, 0)
        assert abs(abs(quat[0, 0]) - 1.0) < 1e-6
        np.testing.assert_allclose(quat[0, 1:], [0, 0, 0], atol=1e-6)

    def test_90deg_x(self):
        r = Rotation.from_euler("x", 90, degrees=True)
        mat = r.as_matrix().reshape(1, 3, 3)
        quat_wxyz = _matrix_to_quaternion(mat)[0]
        # Convert back and compare
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        r_back = Rotation.from_quat(quat_xyzw)
        np.testing.assert_allclose(r_back.as_matrix(), r.as_matrix(), atol=1e-6)

    def test_roundtrip(self):
        """rot6d → matrix → quaternion → matrix should be identity transform."""
        rotations = Rotation.random(15, random_state=99)
        for r in rotations:
            rot6d = _rot6d_from_scipy(r).reshape(1, 6)
            mat = _rot6d_to_matrix(rot6d)
            quat_wxyz = _matrix_to_quaternion(mat)[0]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            r_back = Rotation.from_quat(quat_xyzw)
            np.testing.assert_allclose(r_back.as_matrix(), r.as_matrix(), atol=1e-5)

    def test_glTF_xyzw_order(self):
        """Verify our wxyz → xyzw conversion matches glTF convention."""
        r = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
        mat = r.as_matrix().reshape(1, 3, 3)
        quat_wxyz = _matrix_to_quaternion(mat)[0]
        # Simulate what export_glb_animation does
        quat_gltf = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        # scipy uses (x,y,z,w) natively
        quat_scipy = r.as_quat()
        np.testing.assert_allclose(quat_gltf, quat_scipy, atol=1e-6)


# ── RPM rest pose data ──────────────────────────────────────────────


class TestRPMRestPose:
    def test_rest_rotations_are_valid(self):
        """All RPM rest quaternions must be unit quaternions."""
        for name, xyzw in RPM_REST_ROTATIONS_XYZW.items():
            norm = np.linalg.norm(xyzw)
            assert abs(norm - 1.0) < 1e-3, f"{name} quaternion norm={norm}"

    def test_rest_matrices_are_rotations(self):
        """Rest rotation matrices must be orthogonal with det=1."""
        for name in RPM_REST_ROTATIONS_XYZW:
            m = _get_rpm_rest_matrix(name)
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-5,
                                       err_msg=f"{name} not orthogonal")
            np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-5,
                                       err_msg=f"{name} det != 1")

    def test_left_right_symmetry(self):
        """Left/Right bones should be mirrored in Y and Z components."""
        pairs = [
            ("LeftShoulder", "RightShoulder"),
            ("LeftArm", "RightArm"),
            ("LeftForeArm", "RightForeArm"),
            ("LeftHand", "RightHand"),
            ("LeftUpLeg", "RightUpLeg"),
            ("LeftLeg", "RightLeg"),
            ("LeftFoot", "RightFoot"),
        ]
        for left, right in pairs:
            lq = np.array(RPM_REST_ROTATIONS_XYZW[left])
            rq = np.array(RPM_REST_ROTATIONS_XYZW[right])
            # X and W should match, Y and Z should be negated
            np.testing.assert_allclose(lq[0], rq[0], atol=1e-3,
                                       err_msg=f"{left}/{right} X mismatch")
            np.testing.assert_allclose(lq[1], -rq[1], atol=1e-3,
                                       err_msg=f"{left}/{right} Y not negated")
            np.testing.assert_allclose(lq[2], -rq[2], atol=1e-3,
                                       err_msg=f"{left}/{right} Z not negated")
            np.testing.assert_allclose(lq[3], rq[3], atol=1e-3,
                                       err_msg=f"{left}/{right} W mismatch")

    def test_world_rest_accumulates_parents(self):
        """World rest for Spine = rest(Hips) @ rest(Spine)."""
        w_spine = _get_rpm_world_rest_matrix("Spine")
        expected = _get_rpm_rest_matrix("Hips") @ _get_rpm_rest_matrix("Spine")
        np.testing.assert_allclose(w_spine, expected, atol=1e-7)

    def test_world_rest_deep_chain(self):
        """World rest for LeftFoot = Hips @ LeftUpLeg @ LeftLeg @ LeftFoot."""
        chain = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot"]
        expected = np.eye(3)
        for name in chain:
            expected = expected @ _get_rpm_rest_matrix(name)
        actual = _get_rpm_world_rest_matrix("LeftFoot")
        np.testing.assert_allclose(actual, expected, atol=1e-7)


# ── Retargeting formula ─────────────────────────────────────────────


class TestRetargetRotation:
    def test_identity_smpl_produces_rest_pose(self):
        """When SMPL rotation is identity, output should be RPM rest rotation."""
        identity = np.eye(3).reshape(1, 3, 3)
        for name in RPM_REST_ROTATIONS_XYZW:
            result = _retarget_rotation(identity, name, RPM_REST_ROTATIONS_XYZW)
            expected = _get_rpm_rest_matrix(name)
            np.testing.assert_allclose(result[0], expected, atol=1e-6,
                                       err_msg=f"{name}: identity SMPL should produce rest pose")

    def test_root_formula_simplification(self):
        """Root bone (Hips) formula: R_smpl @ R_rest (no parent conjugation)."""
        r = Rotation.from_euler("y", 45, degrees=True).as_matrix().reshape(1, 3, 3)
        result = _retarget_rotation(r, "Hips", RPM_REST_ROTATIONS_XYZW)
        expected = r[0] @ _get_rpm_rest_matrix("Hips")
        np.testing.assert_allclose(result[0], expected, atol=1e-7)

    def test_child_uses_parent_conjugation(self):
        """Non-root bones must conjugate by parent world rest rotation."""
        r = Rotation.from_euler("x", 30, degrees=True).as_matrix().reshape(1, 3, 3)
        result = _retarget_rotation(r, "Spine", RPM_REST_ROTATIONS_XYZW)

        wp = _get_rpm_world_rest_matrix("Hips")  # parent of Spine
        rest_local = _get_rpm_rest_matrix("Spine")
        expected = wp.T @ r[0] @ wp @ rest_local
        np.testing.assert_allclose(result[0], expected, atol=1e-7)

    def test_output_is_valid_rotation(self):
        """Retargeted matrices must be proper rotations."""
        rotations = Rotation.random(5, random_state=42)
        smpl_rots = np.array([r.as_matrix() for r in rotations])
        for name in ["Hips", "LeftArm", "RightLeg", "Spine2"]:
            result = _retarget_rotation(smpl_rots, name, RPM_REST_ROTATIONS_XYZW)
            for i in range(len(result)):
                np.testing.assert_allclose(result[i] @ result[i].T, np.eye(3),
                                           atol=1e-5, err_msg=f"{name} frame {i}")


# ── Full retarget pipeline ──────────────────────────────────────────


class TestRetargetToMixamo:
    def test_identity_produces_all_bones(self):
        rot6d, transl = _make_smpl_frames(5)
        result = retarget_to_mixamo(rot6d, transl, fps=30)
        assert result["fps"] == 30
        assert result["num_frames"] == 5
        # Should have all mapped bones
        assert "Hips" in result["bones"]
        assert "LeftArm" in result["bones"]
        assert "RightFoot" in result["bones"]
        assert len(result["bones"]) == len(SMPL_TO_MIXAMO)

    def test_root_has_position(self):
        rot6d, transl = _make_smpl_frames(3)
        result = retarget_to_mixamo(rot6d, transl, fps=30)
        assert result["bones"]["Hips"]["position"] is not None
        np.testing.assert_allclose(result["bones"]["Hips"]["position"][:, 1], 0.9, atol=1e-7)

    def test_non_root_has_no_position(self):
        rot6d, transl = _make_smpl_frames(3)
        result = retarget_to_mixamo(rot6d, transl, fps=30)
        assert result["bones"]["Spine"]["position"] is None
        assert result["bones"]["LeftArm"]["position"] is None

    def test_zero_root_xz(self):
        rot6d, transl = _make_smpl_frames(3)
        transl[:, 0] = 1.0  # X movement
        transl[:, 2] = 2.0  # Z movement
        result = retarget_to_mixamo(rot6d, transl, fps=30, zero_root_xz=True)
        pos = result["bones"]["Hips"]["position"]
        np.testing.assert_allclose(pos[:, 0], 0.0, atol=1e-7)  # X zeroed
        np.testing.assert_allclose(pos[:, 2], 0.0, atol=1e-7)  # Z zeroed
        np.testing.assert_allclose(pos[:, 1], 0.9, atol=1e-7)  # Y preserved

    def test_scale_factor(self):
        rot6d, transl = _make_smpl_frames(3)
        result = retarget_to_mixamo(rot6d, transl, fps=30, scale=100.0)
        pos = result["bones"]["Hips"]["position"]
        np.testing.assert_allclose(pos[:, 1], 90.0, atol=1e-5)  # 0.9 * 100

    def test_wave_animation_only_moves_right_arm(self):
        """Wave animation should primarily affect RightArm and RightForeArm."""
        rot6d, transl = _make_wave_animation(30)
        result = retarget_to_mixamo(rot6d, transl, fps=30)

        # RightArm rest pose (from identity SMPL input)
        rest_rot6d, rest_transl = _make_smpl_frames(1)
        rest_result = retarget_to_mixamo(rest_rot6d, rest_transl, fps=30)
        rest_right_arm = rest_result["bones"]["RightArm"]["rotations"][0]

        # RightArm should deviate from rest at some frames (peak of wave)
        right_arm_rots = result["bones"]["RightArm"]["rotations"]
        max_deviation = max(
            np.max(np.abs(right_arm_rots[i] - rest_right_arm))
            for i in range(len(right_arm_rots))
        )
        assert max_deviation > 0.01, f"RightArm should be animated, max_dev={max_deviation}"

        # LeftArm should stay at rest (constant across all frames)
        left_arm_rots = result["bones"]["LeftArm"]["rotations"]
        left_arm_var = np.max(np.std(left_arm_rots, axis=0))
        assert left_arm_var < 1e-6, "LeftArm should not be animated"


# ── Export formats ───────────────────────────────────────────────────


class TestExportFormats:
    @pytest.fixture
    def wave_retargeted(self):
        rot6d, transl = _make_wave_animation(10)
        return retarget_to_mixamo(rot6d, transl, fps=30)

    def test_bvh_export(self, wave_retargeted, tmp_path):
        path = str(tmp_path / "test.bvh")
        export_bvh(wave_retargeted, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "HIERARCHY" in content
        assert "MOTION" in content
        assert "Frames: 10" in content
        assert "ROOT Hips" in content
        assert "JOINT Spine" in content

    def test_glb_export(self, wave_retargeted, tmp_path):
        path = str(tmp_path / "test.glb")
        export_glb_animation(wave_retargeted, path)
        assert os.path.exists(path)
        # GLB starts with magic bytes 'glTF'
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF"

    def test_gltf_export(self, wave_retargeted, tmp_path):
        path = str(tmp_path / "test.gltf")
        export_gltf_animation(wave_retargeted, path)
        assert os.path.exists(path)
        # glTF is valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "animations" in data
        assert "accessors" in data

    def test_gltf_has_all_bones(self, wave_retargeted, tmp_path):
        path = str(tmp_path / "test.gltf")
        export_gltf_animation(wave_retargeted, path)
        with open(path) as f:
            data = json.load(f)
        node_names = {n["name"] for n in data["nodes"]}
        assert "Hips" in node_names
        assert "RightArm" in node_names
        assert "LeftFoot" in node_names

    def test_gltf_has_animation_channels(self, wave_retargeted, tmp_path):
        path = str(tmp_path / "test.gltf")
        export_gltf_animation(wave_retargeted, path)
        with open(path) as f:
            data = json.load(f)
        anim = data["animations"][0]
        assert anim["name"] == "generated"
        # Should have rotation channels for each bone + translation for Hips
        n_bones = len(wave_retargeted["bones"])
        # rotation channels = n_bones, translation channels = 1 (Hips only)
        assert len(anim["channels"]) == n_bones + 1

    def test_glb_gltf_same_quaternions(self, wave_retargeted, tmp_path):
        """GLB and glTF exports must contain identical quaternion data."""
        glb_path = str(tmp_path / "test.glb")
        gltf_path = str(tmp_path / "test.gltf")
        export_glb_animation(wave_retargeted, glb_path)
        export_gltf_animation(wave_retargeted, gltf_path)

        # Parse glTF JSON and extract quaternion data from base64 buffers
        import base64

        with open(gltf_path) as f:
            gltf_data = json.load(f)

        # Decode the base64 buffer
        buf_uri = gltf_data["buffers"][0]["uri"]
        assert buf_uri.startswith("data:application/octet-stream;base64,")
        b64_data = buf_uri.split(",", 1)[1]
        gltf_buffer = base64.b64decode(b64_data)

        # Parse GLB buffer
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()
        # GLB header: magic(4) + version(4) + length(4) = 12 bytes
        # Chunk 0 (JSON): length(4) + type(4) + data
        json_chunk_len = struct.unpack_from("<I", glb_bytes, 12)[0]
        # Chunk 1 (BIN): length(4) + type(4) + data
        bin_offset = 12 + 8 + json_chunk_len
        bin_chunk_len = struct.unpack_from("<I", glb_bytes, bin_offset)[0]
        glb_buffer = glb_bytes[bin_offset + 8: bin_offset + 8 + bin_chunk_len]

        # Compare rotation accessor data (skip timestamp accessor at index 0)
        for acc_idx in range(1, len(gltf_data["accessors"])):
            acc = gltf_data["accessors"][acc_idx]
            bv = gltf_data["bufferViews"][acc["bufferView"]]
            offset = bv["byteOffset"]
            length = bv["byteLength"]

            gltf_chunk = gltf_buffer[offset:offset + length]
            glb_chunk = glb_buffer[offset:offset + length]
            assert gltf_chunk == glb_chunk, f"Accessor {acc_idx} data mismatch"

    def test_gltf_quaternion_continuity(self, tmp_path):
        """Quaternions must be continuous — no sign flips between adjacent frames.

        A backflip is a 360° rotation around X. Without continuity enforcement,
        q and -q (same orientation) cause SLERP to take the wrong path.
        """
        import base64

        # Create a 360° rotation over 30 frames (backflip around X axis)
        n_frames = 30
        rot6d = np.zeros((n_frames, 22, 6), dtype=np.float64)
        transl = np.zeros((n_frames, 3), dtype=np.float64)

        for f in range(n_frames):
            angle = 2 * np.pi * f / (n_frames - 1)  # 0 to 2π
            for j in range(22):
                if j == 0:
                    # Root: full backflip around X
                    mat = Rotation.from_euler("X", angle).as_matrix()
                else:
                    mat = np.eye(3)
                # rot6d is row-major [r00,r01, r10,r11, r20,r21]
                rot6d[f, j, 0] = mat[0, 0]; rot6d[f, j, 1] = mat[0, 1]
                rot6d[f, j, 2] = mat[1, 0]; rot6d[f, j, 3] = mat[1, 1]
                rot6d[f, j, 4] = mat[2, 0]; rot6d[f, j, 5] = mat[2, 1]

        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        path = str(tmp_path / "backflip.gltf")
        export_gltf_animation(retargeted, path)

        with open(path) as f_in:
            data = json.load(f_in)

        buf_uri = data["buffers"][0]["uri"]
        buf_bytes = base64.b64decode(buf_uri.split(",", 1)[1])

        # Check all VEC4 (quaternion) accessors for continuity
        for acc in data["accessors"]:
            if acc["type"] != "VEC4":
                continue
            bv = data["bufferViews"][acc["bufferView"]]
            offset = bv["byteOffset"]
            count = acc["count"]
            quats = np.frombuffer(buf_bytes, dtype=np.float32,
                                  count=count * 4, offset=offset).reshape(-1, 4)
            # Adjacent quaternions should have positive dot product (same hemisphere)
            for i in range(1, len(quats)):
                dot = np.dot(quats[i], quats[i - 1])
                assert dot >= -0.01, (
                    f"Quaternion discontinuity at frame {i}: "
                    f"dot={dot:.4f}, q[{i}]={quats[i]}, q[{i-1}]={quats[i-1]}"
                )

    def test_backflip_rotation_progresses_smoothly(self, tmp_path):
        """Backflip rotation must progress monotonically — no snapping at 180°.

        When the character reaches upside-down (180°), the retargeted Hips
        rotation should continue smoothly, not flip or reverse direction.
        """
        import base64

        n_frames = 60
        rot6d = np.zeros((n_frames, 22, 6), dtype=np.float64)
        transl = np.zeros((n_frames, 3), dtype=np.float64)

        input_angles = []
        for f in range(n_frames):
            angle = 2 * np.pi * f / (n_frames - 1)  # 0 to 2π
            input_angles.append(np.degrees(angle))
            for j in range(22):
                if j == 0:
                    mat = Rotation.from_euler("X", angle).as_matrix()
                else:
                    mat = np.eye(3)
                rot6d[f, j, 0] = mat[0, 0]; rot6d[f, j, 1] = mat[0, 1]
                rot6d[f, j, 2] = mat[1, 0]; rot6d[f, j, 3] = mat[1, 1]
                rot6d[f, j, 4] = mat[2, 0]; rot6d[f, j, 5] = mat[2, 1]

        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        path = str(tmp_path / "backflip.gltf")
        export_gltf_animation(retargeted, path)

        with open(path) as f_in:
            data = json.load(f_in)

        buf_uri = data["buffers"][0]["uri"]
        buf_bytes = base64.b64decode(buf_uri.split(",", 1)[1])

        # Find the Hips node and its rotation channel
        hips_node_idx = None
        for i, node in enumerate(data["nodes"]):
            if node["name"] == "Hips":
                hips_node_idx = i
                break
        assert hips_node_idx is not None

        hips_rot_acc_idx = None
        for ch in data["animations"][0]["channels"]:
            if ch["target"]["node"] == hips_node_idx and ch["target"]["path"] == "rotation":
                sampler = data["animations"][0]["samplers"][ch["sampler"]]
                hips_rot_acc_idx = sampler["output"]
                break
        assert hips_rot_acc_idx is not None

        acc = data["accessors"][hips_rot_acc_idx]
        bv = data["bufferViews"][acc["bufferView"]]
        quats = np.frombuffer(buf_bytes, dtype=np.float32,
                              count=acc["count"] * 4, offset=bv["byteOffset"]).reshape(-1, 4)
        # glTF format: (x, y, z, w)

        # Convert exported quaternions back to rotation angles around X
        # For a rotation Rx(θ): quat = (sin(θ/2), 0, 0, cos(θ/2))
        # The x component contains the rotation info, but rest pose offsets it
        exported_rots = Rotation.from_quat(quats)  # scipy takes (x,y,z,w)

        # Check that angular velocity between adjacent frames never reverses
        # (angle between frame[i] and frame[i+1] should always be positive/forward)
        max_frame_delta = 0
        for i in range(1, len(quats)):
            # Angular distance between consecutive frames
            r_diff = exported_rots[i] * exported_rots[i - 1].inv()
            angle = r_diff.magnitude()  # always positive, 0 to π
            max_frame_delta = max(max_frame_delta, np.degrees(angle))

        # Each frame should be ~6° apart (360°/60 frames), allow up to 25°
        # If there's a flip, we'd see a ~180° jump
        assert max_frame_delta < 25.0, (
            f"Backflip has discontinuity: max frame delta = {max_frame_delta:.1f}° "
            f"(expected ~6° per frame)"
        )

    def test_gltf_quaternions_are_unit(self, wave_retargeted, tmp_path):
        """All quaternions in the exported glTF must be unit quaternions."""
        import base64

        path = str(tmp_path / "test.gltf")
        export_gltf_animation(wave_retargeted, path)

        with open(path) as f:
            data = json.load(f)

        buf_uri = data["buffers"][0]["uri"]
        buf_bytes = base64.b64decode(buf_uri.split(",", 1)[1])

        for acc in data["accessors"]:
            if acc["type"] != "VEC4":
                continue
            bv = data["bufferViews"][acc["bufferView"]]
            offset = bv["byteOffset"]
            count = acc["count"]
            quats = np.frombuffer(buf_bytes, dtype=np.float32,
                                  count=count * 4, offset=offset).reshape(-1, 4)
            norms = np.linalg.norm(quats, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4,
                                       err_msg="Non-unit quaternion found in export")


# ── Hierarchy consistency ────────────────────────────────────────────


class TestHierarchy:
    def test_parent_lookup_complete(self):
        """Every bone in hierarchy should have a parent entry."""
        for bone in MIXAMO_HIERARCHY:
            assert bone in MIXAMO_PARENT

    def test_parent_lookup_consistent(self):
        """Parent lookup must be inverse of hierarchy children."""
        for parent, children in MIXAMO_HIERARCHY.items():
            for child in children:
                assert MIXAMO_PARENT[child] == parent

    def test_smpl_mapping_covers_hierarchy(self):
        """Every SMPL-mapped bone should exist in the hierarchy."""
        for _, name in SMPL_TO_MIXAMO.items():
            assert name in MIXAMO_HIERARCHY, f"{name} not in hierarchy"


# ── Custom rig support ──────────────────────────────────────────────


def _make_minimal_glb(bone_quats: dict[str, tuple]) -> bytes:
    """Create a minimal valid GLB with bone nodes and their rest quaternions."""
    nodes = []
    for name, quat in bone_quats.items():
        nodes.append({"name": name, "rotation": list(quat)})

    gltf_json = json.dumps({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
    }).encode("utf-8")

    # Pad JSON to 4-byte alignment
    json_padding = (4 - len(gltf_json) % 4) % 4
    gltf_json += b" " * json_padding

    # GLB header: magic(4) + version(4) + totalLength(4)
    # JSON chunk: length(4) + type(4) + data
    total_len = 12 + 8 + len(gltf_json)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    json_chunk_header = struct.pack("<II", len(gltf_json), 0x4E4F534A)

    return header + json_chunk_header + gltf_json


class TestCustomRig:
    def test_extract_rest_from_glb_basic(self):
        """Extract rest quaternions from a synthetic GLB."""
        quats = {"Hips": (0, 0, 0, 1), "Spine": (0.1, 0, 0, 0.995)}
        glb = _make_minimal_glb(quats)
        result = extract_rest_from_glb(glb)
        assert "Hips" in result
        assert "Spine" in result
        np.testing.assert_allclose(result["Hips"], (0, 0, 0, 1))
        np.testing.assert_allclose(result["Spine"], (0.1, 0, 0, 0.995))

    def test_extract_strips_mixamorig_prefix(self):
        """Bone names with 'mixamorig:' prefix should be stripped."""
        quats = {"mixamorig:Hips": (0, 0, 0, 1), "mixamorig:LeftArm": (0.1, 0.2, 0.3, 0.9)}
        glb = _make_minimal_glb(quats)
        result = extract_rest_from_glb(glb)
        assert "Hips" in result
        assert "LeftArm" in result

    def test_extract_ignores_non_bone_nodes(self):
        """Nodes that aren't Mixamo bones should be ignored."""
        quats = {"Hips": (0, 0, 0, 1), "MeshBody": (0, 0, 0, 1), "Camera": (0, 0, 0, 1)}
        glb = _make_minimal_glb(quats)
        result = extract_rest_from_glb(glb)
        assert "Hips" in result
        assert "MeshBody" not in result
        assert "Camera" not in result

    def test_extract_falls_back_on_empty_glb(self):
        """GLB with no Mixamo bones should return RPM defaults."""
        quats = {"MeshBody": (0, 0, 0, 1)}
        glb = _make_minimal_glb(quats)
        result = extract_rest_from_glb(glb)
        assert result is RPM_REST_ROTATIONS_XYZW

    def test_retarget_with_custom_rest(self):
        """retarget_to_mixamo should accept custom rest_rotations."""
        rot6d, transl = _make_smpl_frames(5)
        identity_rest = {bone: (0, 0, 0, 1) for bone in MIXAMO_HIERARCHY}
        result = retarget_to_mixamo(rot6d, transl, rest_rotations=identity_rest)
        assert len(result["bones"]) == 20

    def test_custom_rest_produces_different_output(self):
        """Custom rest rotations should produce different results than RPM defaults."""
        rot6d, transl = _make_smpl_frames(5)
        # RPM defaults
        rpm_result = retarget_to_mixamo(rot6d, transl)
        # Identity rest (all bones at identity)
        identity_rest = {bone: (0, 0, 0, 1) for bone in MIXAMO_HIERARCHY}
        custom_result = retarget_to_mixamo(rot6d, transl, rest_rotations=identity_rest)
        # With identity SMPL input and identity rest, output should be identity
        hips_custom = custom_result["bones"]["Hips"]["rotations"][0]
        np.testing.assert_allclose(hips_custom, np.eye(3), atol=1e-6)
        # With identity SMPL input and RPM rest, output should NOT be identity
        hips_rpm = rpm_result["bones"]["Hips"]["rotations"][0]
        assert not np.allclose(hips_rpm, np.eye(3), atol=1e-3)


# ── Looping crossfade ─────────────────────────────────────────────


class TestMakeLoopable:
    def test_loop_does_not_change_frame_count(self):
        """make_loopable should not add or remove frames."""
        rot6d, transl = _make_wave_animation(60)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        original_count = retargeted["num_frames"]
        make_loopable(retargeted)
        assert retargeted["num_frames"] == original_count

    def test_loop_blends_last_frames_toward_first(self):
        """After looping, the last frame should be close to the first frame."""
        rot6d, transl = _make_wave_animation(60)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)

        # Before looping: last frame differs from first
        hips_rots = retargeted["bones"]["Hips"]["rotations"]
        diff_before = np.max(np.abs(hips_rots[-1] - hips_rots[0]))

        make_loopable(retargeted, blend_frames=15)

        hips_rots_after = retargeted["bones"]["Hips"]["rotations"]
        diff_after = np.max(np.abs(hips_rots_after[-1] - hips_rots_after[0]))
        assert diff_after < diff_before or diff_after < 0.05

    def test_loop_blends_root_position(self):
        """Root translation should also blend toward frame 0."""
        rot6d, transl = _make_smpl_frames(60)
        # Add some root movement
        transl[:, 0] = np.linspace(0, 2.0, 60)  # X drift
        transl[:, 2] = np.linspace(0, 1.0, 60)  # Z drift

        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        pos_before = retargeted["bones"]["Hips"]["position"][-1].copy()

        make_loopable(retargeted, blend_frames=15)

        pos_after = retargeted["bones"]["Hips"]["position"][-1]
        pos_first = retargeted["bones"]["Hips"]["position"][0]
        # Last frame should be closer to first frame after looping
        dist_before = np.linalg.norm(pos_before - pos_first)
        dist_after = np.linalg.norm(pos_after - pos_first)
        assert dist_after < dist_before

    def test_loop_wrap_is_smooth(self):
        """Angular delta at the wrap point (last→first) should be small."""
        rot6d, transl = _make_wave_animation(90)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        make_loopable(retargeted, blend_frames=20)

        # Check wrap smoothness for several bones
        for bone_name in ["Hips", "RightArm", "Spine"]:
            rots = retargeted["bones"][bone_name]["rotations"]
            r_last = Rotation.from_matrix(rots[-1])
            r_first = Rotation.from_matrix(rots[0])
            wrap_delta = (r_first * r_last.inv()).magnitude()
            wrap_deg = np.degrees(wrap_delta)
            # Should be similar to the inter-frame delta, not a big jump
            assert wrap_deg < 10.0, (
                f"{bone_name}: wrap delta = {wrap_deg:.1f}° (should be < 10°)"
            )

    def test_loop_euler_recomputed(self):
        """Euler angles should be updated after blending."""
        rot6d, transl = _make_wave_animation(60)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        make_loopable(retargeted, blend_frames=15)

        # Verify euler_zxy matches the blended rotation matrices
        for bone_name, data in retargeted["bones"].items():
            expected_euler = _matrix_to_euler(data["rotations"], "ZXY")
            np.testing.assert_allclose(
                data["euler_zxy"], expected_euler, atol=1e-5,
                err_msg=f"{bone_name}: euler_zxy not recomputed after loop blend"
            )

    def test_loop_too_short_is_noop(self):
        """Animations too short for blending should be returned unchanged."""
        rot6d, transl = _make_smpl_frames(4)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        original_hips = retargeted["bones"]["Hips"]["rotations"].copy()
        make_loopable(retargeted, blend_frames=15)
        np.testing.assert_array_equal(
            retargeted["bones"]["Hips"]["rotations"], original_hips
        )

    def test_loop_glb_export_quaternion_continuity(self, tmp_path):
        """Looped animation should still have continuous quaternions in GLB export."""
        import base64

        rot6d, transl = _make_wave_animation(60)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30)
        make_loopable(retargeted, blend_frames=15)

        path = str(tmp_path / "loop.gltf")
        export_gltf_animation(retargeted, path)

        with open(path) as f:
            data = json.load(f)

        buf_uri = data["buffers"][0]["uri"]
        buf_bytes = base64.b64decode(buf_uri.split(",", 1)[1])

        for acc in data["accessors"]:
            if acc["type"] != "VEC4":
                continue
            bv = data["bufferViews"][acc["bufferView"]]
            quats = np.frombuffer(buf_bytes, dtype=np.float32,
                                  count=acc["count"] * 4, offset=bv["byteOffset"]).reshape(-1, 4)
            for i in range(1, len(quats)):
                dot = np.dot(quats[i], quats[i - 1])
                assert dot >= -0.01, (
                    f"Quaternion discontinuity at frame {i}: dot={dot:.4f}"
                )


# ── Real avatar GLB integration ──────────────────────────────────

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_AVATAR_GLB = os.path.join(FIXTURE_DIR, "test_avatar.glb")


@pytest.mark.skipif(not os.path.exists(REAL_AVATAR_GLB), reason="test_avatar.glb fixture not present")
class TestRealAvatarGLB:
    @pytest.fixture
    def avatar_glb_bytes(self):
        with open(REAL_AVATAR_GLB, "rb") as f:
            return f.read()

    def test_extract_rest_finds_bones(self, avatar_glb_bytes):
        """Real RPM avatar GLB should contain Mixamo bones."""
        rest = extract_rest_from_glb(avatar_glb_bytes)
        assert "Hips" in rest
        assert "Spine" in rest
        assert "LeftArm" in rest
        assert "RightFoot" in rest
        # Should find most bones
        assert len(rest) >= 15, f"Only found {len(rest)} bones, expected >= 15"

    def test_extracted_quats_are_unit(self, avatar_glb_bytes):
        """Extracted quaternions should be unit quaternions."""
        rest = extract_rest_from_glb(avatar_glb_bytes)
        for name, xyzw in rest.items():
            norm = np.linalg.norm(xyzw)
            assert abs(norm - 1.0) < 1e-3, f"{name}: quaternion norm={norm}"

    def test_extracted_quats_close_to_rpm_defaults(self, avatar_glb_bytes):
        """Extracted rest quats should be reasonably close to hardcoded RPM defaults.

        Not all RPM avatars share identical rest poses — arm/leg angles vary
        between body types. Spine/torso bones should match closely, while
        extremities can differ more.
        """
        rest = extract_rest_from_glb(avatar_glb_bytes)
        close_count = 0
        total = 0
        for name in RPM_REST_ROTATIONS_XYZW:
            if name not in rest:
                continue
            total += 1
            extracted = np.array(rest[name])
            hardcoded = np.array(RPM_REST_ROTATIONS_XYZW[name])
            dot = abs(np.dot(extracted, hardcoded))
            if dot > 0.95:
                close_count += 1
        # At least half the bones should match closely
        assert close_count >= total // 2, (
            f"Only {close_count}/{total} bones match RPM defaults (expected >= {total // 2})"
        )

    def test_retarget_with_real_avatar(self, avatar_glb_bytes):
        """Full pipeline: extract rest from real GLB → retarget → export."""
        rest = extract_rest_from_glb(avatar_glb_bytes)
        rot6d, transl = _make_wave_animation(30)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30, rest_rotations=rest)
        assert retargeted["num_frames"] == 30
        assert len(retargeted["bones"]) == 20

    def test_retarget_and_export_with_real_avatar(self, avatar_glb_bytes, tmp_path):
        """End-to-end: real avatar rest → retarget → GLB export."""
        rest = extract_rest_from_glb(avatar_glb_bytes)
        rot6d, transl = _make_wave_animation(30)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30, rest_rotations=rest)

        glb_path = str(tmp_path / "real_avatar.glb")
        export_glb_animation(retargeted, glb_path)
        assert os.path.exists(glb_path)

        with open(glb_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF"

    def test_retarget_loop_with_real_avatar(self, avatar_glb_bytes, tmp_path):
        """End-to-end: real avatar rest → retarget → loop → GLB export."""
        rest = extract_rest_from_glb(avatar_glb_bytes)
        rot6d, transl = _make_wave_animation(60)
        retargeted = retarget_to_mixamo(rot6d, transl, fps=30, rest_rotations=rest)
        make_loopable(retargeted, blend_frames=15)

        # Check wrap smoothness
        hips_rots = retargeted["bones"]["Hips"]["rotations"]
        r_last = Rotation.from_matrix(hips_rots[-1])
        r_first = Rotation.from_matrix(hips_rots[0])
        wrap_deg = np.degrees((r_first * r_last.inv()).magnitude())
        assert wrap_deg < 10.0, f"Hips wrap delta = {wrap_deg:.1f}°"

        glb_path = str(tmp_path / "real_avatar_loop.glb")
        export_glb_animation(retargeted, glb_path)
        assert os.path.exists(glb_path)
