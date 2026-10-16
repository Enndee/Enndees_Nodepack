"""Unit tests for tools/merge_ply_splats.py (synthetic PLYs, no Studio needed)."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "tools"))

import merge_ply_splats as merge  # noqa: E402

PROPS = (("float", "x"), ("float", "y"), ("float", "z"), ("float", "opacity"))


def write_ply(path, rows, props=PROPS, file_format="binary_little_endian 1.0"):
    header = ["ply", f"format {file_format}", f"element vertex {len(rows)}"]
    header += [f"property {kind} {name}" for kind, name in props]
    header.append("end_header")
    path = Path(path)
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<" + "f" * len(row), *row))


def read_rows(path):
    header = merge.parse_ply_header(path)
    width = len(header["properties"])
    with Path(path).open("rb") as handle:
        handle.seek(header["header_bytes"])
        data = handle.read()
    return [struct.unpack("<" + "f" * width, data[i * header["row_bytes"]:
                                                   (i + 1) * header["row_bytes"]])
            for i in range(header["count"])]


class MergePlyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_merge_concatenates_rows_and_counts(self):
        first = self.dir / "subject.ply"
        second = self.dir / "background.ply"
        out = self.dir / "merged.ply"
        write_ply(first, [(1.0, 2.0, 3.0, 0.9), (4.0, 5.0, 6.0, 0.8)])
        write_ply(second, [(7.0, 8.0, 9.0, 0.7)])
        counts = merge.merge_ply([first, second], out)
        self.assertEqual(counts, [2, 1])
        rows = read_rows(out)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row[0] for row in rows], [1.0, 4.0, 7.0])
        header = merge.parse_ply_header(out)
        self.assertEqual(header["count"], 3)

    def test_the_output_keeps_the_property_layout(self):
        first = self.dir / "a.ply"
        second = self.dir / "b.ply"
        out = self.dir / "merged.ply"
        write_ply(first, [(1.0, 2.0, 3.0, 0.9)])
        write_ply(second, [(4.0, 5.0, 6.0, 0.8)])
        merge.merge_ply([first, second], out)
        self.assertEqual(merge.parse_ply_header(out)["properties"],
                         merge.parse_ply_header(first)["properties"])

    def test_property_mismatch_is_rejected(self):
        first = self.dir / "a.ply"
        second = self.dir / "b.ply"
        other = (("float", "x"), ("float", "y"), ("float", "z"),
                 ("float", "opacity"), ("float", "scale_0"))
        write_ply(first, [(1.0, 2.0, 3.0, 0.9)])
        write_ply(second, [(1.0, 2.0, 3.0, 0.9, 0.1)], props=other)
        with self.assertRaises(ValueError):
            merge.merge_ply([first, second], self.dir / "merged.ply")

    def test_truncated_files_are_rejected(self):
        broken = self.dir / "broken.ply"
        write_ply(broken, [(1.0, 2.0, 3.0, 0.9)])
        data = broken.read_bytes()
        broken.write_bytes(data[:-3])
        with self.assertRaises(ValueError):
            merge.parse_ply_header(broken)

    def test_ascii_ply_is_rejected(self):
        ascii_ply = self.dir / "ascii.ply"
        write_ply(ascii_ply, [(1.0, 2.0, 3.0, 0.9)], file_format="ascii 1.0")
        with self.assertRaises(ValueError):
            merge.parse_ply_header(ascii_ply)

    def test_output_may_not_be_one_of_the_inputs(self):
        first = self.dir / "a.ply"
        write_ply(first, [(1.0, 2.0, 3.0, 0.9)])
        with self.assertRaises(ValueError):
            merge.merge_ply([first, first], first)

    def test_info_reports_counts(self):
        ply = self.dir / "a.ply"
        write_ply(ply, [(1.0, 2.0, 3.0, 0.9), (4.0, 5.0, 6.0, 0.8)])
        self.assertEqual(merge.main(["--info", str(ply)]), 0)


if __name__ == "__main__":
    unittest.main()