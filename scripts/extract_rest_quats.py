"""Extract bone rest quaternions from an RPM avatar GLB file.

Usage:
    python scripts/extract_rest_quats.py <avatar.glb>
    python scripts/extract_rest_quats.py <url>

Prints bone quaternions in (x, y, z, w) format for comparison with retargeter.py.
"""

import json
import struct
import sys
import urllib.request

BONES_OF_INTEREST = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
]


def load_glb(path_or_url: str) -> tuple[dict, bytes]:
    """Load a GLB file and return (JSON chunk, BIN chunk)."""
    if path_or_url.startswith("http"):
        print(f"Downloading {path_or_url}...", file=sys.stderr)
        req = urllib.request.Request(path_or_url)
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
    else:
        with open(path_or_url, "rb") as f:
            data = f.read()

    # Parse GLB header
    magic, version, total_len = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"Not a GLB file (magic={magic:#x})"

    # Chunk 0: JSON
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A, "First chunk is not JSON"
    json_data = json.loads(data[20 : 20 + json_len])

    # Chunk 1: BIN (optional)
    bin_offset = 12 + 8 + json_len
    bin_data = b""
    if bin_offset < len(data):
        bin_len, bin_type = struct.unpack_from("<II", data, bin_offset)
        bin_data = data[bin_offset + 8 : bin_offset + 8 + bin_len]

    return json_data, bin_data


def extract_bone_quats(gltf: dict) -> dict[str, tuple[float, float, float, float]]:
    """Extract bone node quaternions from glTF JSON."""
    result = {}
    for node in gltf.get("nodes", []):
        name = node.get("name", "")
        # Strip common prefixes
        clean = name.replace("mixamorig:", "").replace("mixamorig", "")
        if clean in BONES_OF_INTEREST:
            rot = node.get("rotation", [0, 0, 0, 1])  # glTF default = identity
            result[clean] = tuple(rot)  # (x, y, z, w)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_rest_quats.py <avatar.glb or URL>", file=sys.stderr)
        sys.exit(1)

    gltf, _ = load_glb(sys.argv[1])
    quats = extract_bone_quats(gltf)

    print(f"Found {len(quats)} bones\n")

    # Print in the format used by retargeter.py
    print("RPM_REST_ROTATIONS_XYZW = {")
    for bone in BONES_OF_INTEREST:
        if bone in quats:
            x, y, z, w = quats[bone]
            print(f'    "{bone}":' + " " * (18 - len(bone)) + f"({x:+8.4f}, {y:+8.4f}, {z:+8.4f}, {w:+8.4f}),")
        else:
            print(f'    # "{bone}" — NOT FOUND')
    print("}")


if __name__ == "__main__":
    main()
