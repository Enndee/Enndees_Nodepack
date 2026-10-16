"""Merge separately trained 3DGS splat PLYs into one file (same world frame).

The subject/background training-view split trains two splats - one supervised on
the subject only, one on the background only - from the SAME COLMAP
reconstruction. Because both inherit the identical poses they already share one
world frame and one scale, so combining them afterwards is a pure byte-level
concatenation of the gaussian rows: no registration, no alignment, no rescale.

    python merge_ply_splats.py merged.ply subject.ply background.ply
    python merge_ply_splats.py --info splat.ply

Supported: ``binary_little_endian`` PLY with a single ``element vertex`` whose
properties are all numeric - the standard 3DGS export (x/y/z, nx/ny/nz, f_dc_*,
f_rest_*, opacity, scale_*, rot_*). All inputs must use the EXACT same property
list (names, types, order), so train both runs with the same ``--sh-degree``.
"""
import argparse
import sys
from pathlib import Path

_TYPE_SIZES = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


def parse_ply_header(path):
    """Header facts for one PLY: format, vertex count, properties, byte offsets.

    Returns ``{"format", "count", "properties", "header_bytes", "row_bytes"}``
    where ``properties`` is a list of ``(type, name)`` in file order. Raises
    ValueError for anything that is not a single-vertex-element binary PLY with
    numeric properties.
    """
    path = Path(path)
    properties = []
    element = None
    count = None
    file_format = None
    header_bytes = 0
    with path.open("rb") as handle:
        first = handle.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        header_bytes += len(first)
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path}: header ended without 'end_header'")
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                file_format = line.split(None, 1)[1]
            elif line.startswith("element "):
                parts = line.split()
                if element is not None:
                    raise ValueError(f"{path}: only one element is supported, "
                                     f"found a second one '{parts[1]}'")
                element, count = parts[1], int(parts[2])
            elif line.startswith("property "):
                parts = line.split()
                if element != "vertex":
                    raise ValueError(f"{path}: properties must belong to the "
                                     f"'vertex' element, found '{element}'")
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break
    if file_format != "binary_little_endian 1.0":
        raise ValueError(f"{path}: unsupported format {file_format!r} - only "
                         f"'binary_little_endian 1.0' can be merged byte-wise")
    if element != "vertex" or not properties:
        raise ValueError(f"{path}: no 'element vertex' with properties found")
    unknown = [kind for kind, _ in properties if kind not in _TYPE_SIZES]
    if unknown:
        raise ValueError(f"{path}: unsupported property type(s) {unknown}")
    row_bytes = sum(_TYPE_SIZES[kind] for kind, _ in properties)
    expected = header_bytes + count * row_bytes
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: truncated or padded - header+data is "
                         f"{expected} bytes but the file has {actual}")
    return {"format": file_format, "count": count, "properties": properties,
            "header_bytes": header_bytes, "row_bytes": row_bytes}


def merge_ply(inputs, output):
    """Concatenate the gaussian rows of ``inputs`` into one PLY at ``output``.

    Returns the list of per-input vertex counts. Inputs must share one property
    layout; ``output`` must not be one of the inputs.
    """
    inputs = [Path(p) for p in inputs]
    output = Path(output)
    if output in inputs:
        raise ValueError("the output path must differ from every input path")
    headers = [parse_ply_header(path) for path in inputs]
    layout = headers[0]["properties"]
    for path, header in zip(inputs, headers):
        if header["properties"] != layout:
            raise ValueError(f"{path}: property layout differs from "
                             f"{inputs[0]} - train both runs with the same "
                             f"settings (e.g. --sh-degree) before merging")
    total = sum(header["count"] for header in headers)
    lines = ["ply", "format binary_little_endian 1.0",
             f"element vertex {total}"]
    lines += [f"property {kind} {name}" for kind, name in layout]
    lines.append("end_header")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(("\n".join(lines) + "\n").encode("ascii"))
        for path, header in zip(inputs, headers):
            with path.open("rb") as src:
                src.seek(header["header_bytes"])
                remaining = header["count"] * header["row_bytes"]
                while remaining:
                    chunk = src.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ValueError(f"{path}: data ended early")
                    out.write(chunk)
                    remaining -= len(chunk)
    return [header["count"] for header in headers]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--info", action="store_true",
                        help="print header facts for the input PLY(s) and exit")
    parser.add_argument("files", nargs="+",
                        help="output.ply in1.ply [in2.ply ...] "
                             "(with --info: just the inputs)")
    args = parser.parse_args(argv)
    if args.info:
        for name in args.files:
            header = parse_ply_header(name)
            print(f"{name}: {header['count']} vertices, "
                  f"{len(header['properties'])} properties, "
                  f"{header['row_bytes']} bytes/row")
        return 0
    if len(args.files) < 3:
        parser.error("need an output path and at least two input PLYs")
    counts = merge_ply(args.files[1:], args.files[0])
    print(f"merged {sum(counts)} gaussians "
          f"({', '.join(str(c) for c in counts)}) -> {args.files[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())