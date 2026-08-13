import tempfile
import unittest
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import zarr
except ImportError:  # Allows source-tree test discovery before dependencies are installed.
    np = None
    zarr = None


ZARR_V3_AVAILABLE = zarr is not None and int(zarr.__version__.split(".", 1)[0]) >= 3
PYTHON_RUNTIME_AVAILABLE = sys.version_info >= (3, 12)


@unittest.skipUnless(PYTHON_RUNTIME_AVAILABLE, "genome-zarr-4bit requires Python 3.12+")
class FastaChunkPlanningTests(unittest.TestCase):
    def test_chunk_planner_splits_long_records_without_changing_record_padding(self):
        """Each parallel job must own one complete packed Zarr chunk."""
        from genome_zarr.codec import encode_4bit
        from genome_zarr.convert import _build_fasta_chunk_tasks, _encode_fasta_pieces, scan_fasta_indexed

        with tempfile.TemporaryDirectory() as directory:
            fasta = Path(directory) / "input.fa"
            fasta.write_text(">chr1\nACGTN\n>chr2\nTTA\n", encoding="ascii")
            records = scan_fasta_indexed(fasta)
            tasks = _build_fasta_chunk_tasks(fasta, records, chunk_bytes=2)

            self.assertEqual([(index, size) for index, size, _ in tasks], [(0, 2), (1, 2), (2, 1)])
            packed = b"".join(
                b"".join(
                    _encode_fasta_pieces(str(fasta), pieces, length).tobytes()
                    for pieces, length in record_parts
                )
                for _, _, record_parts in tasks
            )
            self.assertEqual(packed, encode_4bit("ACGTN") + encode_4bit("TTA"))


@unittest.skipUnless(ZARR_V3_AVAILABLE, "Zarr v3 is not installed")
class ConversionTests(unittest.TestCase):
    def test_full_round_trip_preserves_packed_genome(self):
        from genome_zarr.codec import decode_4bit
        from genome_zarr.convert import compress_zarr, decompress_zarr, fasta_to_zstd, store_statistics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fa"
            fasta.write_text(">chr1 description\nACGTN\nRY\n>chr2\nTTA\n", encoding="ascii")
            # A two-byte chunk forces multiple writer flushes, exercising the
            # reusable bytearray path that previously raised BufferError.
            compressed = fasta_to_zstd(fasta, root / "compressed.zarr", chunk_bases=4, num_workers=2)
            uncompressed = decompress_zarr(compressed, root / "plain.zarr")
            recompressed = compress_zarr(uncompressed, root / "recompressed.zarr")

            first = zarr.open_group(str(compressed), mode="r")
            plain = zarr.open_group(str(uncompressed), mode="r")
            last = zarr.open_group(str(recompressed), mode="r")
            with (compressed / "manifest.json").open("rt", encoding="utf-8") as handle:
                first_manifest = json.load(handle)
            with (uncompressed / "manifest.json").open("rt", encoding="utf-8") as handle:
                plain_manifest = json.load(handle)
            with (recompressed / "manifest.json").open("rt", encoding="utf-8") as handle:
                last_manifest = json.load(handle)

            self.assertEqual(first.attrs["compressor"], "zstd")
            self.assertEqual(first.metadata.zarr_format, 3)
            self.assertEqual(plain.metadata.zarr_format, 3)
            self.assertEqual(last.metadata.zarr_format, 3)
            self.assertEqual(plain.attrs["compressor"], "none")
            self.assertEqual(last.attrs["compressor"], "zstd")
            self.assertEqual(list(first.array_keys()), ["packed_sequence"])
            self.assertEqual(list(plain.array_keys()), ["packed_sequence"])
            self.assertEqual(list(last.array_keys()), ["packed_sequence"])

            first_records = {record["name"]: record for record in first_manifest["records"]}
            plain_records = {record["name"]: record for record in plain_manifest["records"]}
            last_records = {record["name"]: record for record in last_manifest["records"]}
            for chromosome, expected in {"chr1": "ACGTNNN", "chr2": "TTA"}.items():
                first_record = first_records[chromosome]
                plain_record = plain_records[chromosome]
                last_record = last_records[chromosome]
                self.assertTrue(np.array_equal(first["packed_sequence"][first_record["byte_offset"]:first_record["byte_offset"] + first_record["byte_length"]], plain["packed_sequence"][plain_record["byte_offset"]:plain_record["byte_offset"] + plain_record["byte_length"]]))
                self.assertTrue(np.array_equal(plain["packed_sequence"][plain_record["byte_offset"]:plain_record["byte_offset"] + plain_record["byte_length"]], last["packed_sequence"][last_record["byte_offset"]:last_record["byte_offset"] + last_record["byte_length"]]))
                self.assertEqual(decode_4bit(bytes(last["packed_sequence"][last_record["byte_offset"]:last_record["byte_offset"] + last_record["byte_length"]]), last_record["logical_length"]), expected)
            stats = store_statistics(recompressed)
            self.assertEqual(stats["chromosomes"], 2)
            self.assertEqual(stats["logical_bases"], 10)
            self.assertEqual(stats["packed_bytes"], 6)
            self.assertEqual(stats["zarr_format"], 3)
            self.assertGreater(stats["apparent_bytes"], 0)

    def test_fasta_headers_with_coordinate_slash_are_sanitized(self):
        from genome_zarr.convert import fasta_to_zstd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fa"
            fasta.write_text(
                ">JH126484.1/153-268 Collinsella tanakaei YIT 12063 genomic scaffold supercont1.18, whole genome shotgun sequence.\nACGT\n",
                encoding="ascii",
            )

            compressed = fasta_to_zstd(fasta, root / "compressed.zarr", chunk_bases=4)
            with (compressed / "manifest.json").open("rt", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertIn("packed_sequence", list(zarr.open_group(str(compressed), mode="r").array_keys()))
            self.assertEqual(manifest["records"][0]["name"], "JH126484.1_153-268")
            self.assertEqual(manifest["records"][0]["logical_length"], 4)

    def test_dataset_reads_base_windows_from_zstd_and_plain_stores(self):
        from genome_zarr.convert import decompress_zarr, fasta_to_zstd
        from genome_zarr.dataloader import ChunkedGenomeZarrDataset, GenomeZarrDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fa"
            fasta.write_text(">chr1\nACGTN\n>chr2\nTTAC\n", encoding="ascii")
            compressed = fasta_to_zstd(fasta, root / "compressed.zarr", chunk_bases=4, num_workers=1)
            plain = decompress_zarr(compressed, root / "plain.zarr", num_workers=1)

            for store in (compressed, plain):
                dataset = GenomeZarrDataset(store, window_size=3, stride=2, return_metadata=True)
                self.assertEqual(dataset.chromosomes, ("chr1", "chr2"))
                self.assertEqual(len(dataset), 3)
                self.assertEqual(dataset[0]["sequence"].tolist(), [0, 1, 2])
                self.assertEqual(dataset[1]["sequence"].tolist(), [2, 3, 4])
                self.assertEqual(dataset[2]["sequence"].tolist(), [3, 3, 0])
                self.assertEqual(dataset[2]["chromosome"], "chr2")
                self.assertEqual(dataset[2]["start"], 0)
            self.assertEqual(GenomeZarrDataset(compressed, window_size=3).compressor, "zstd")
            self.assertEqual(GenomeZarrDataset(plain, window_size=3).compressor, "none")
            chunked = ChunkedGenomeZarrDataset(compressed, window_size=3, stride=2)
            self.assertEqual([window.tolist() for window in chunked], [[0, 1, 2], [2, 3, 4], [3, 3, 0]])


if __name__ == "__main__":
    unittest.main()
