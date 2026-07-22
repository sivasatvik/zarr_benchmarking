import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import zarr
except ImportError:  # Allows source-tree test discovery before dependencies are installed.
    np = None
    zarr = None


@unittest.skipUnless(zarr is not None, "zarr is not installed")
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
            compressed = fasta_to_zstd(fasta, root / "compressed.zarr", chunk_bases=4)
            uncompressed = decompress_zarr(compressed, root / "plain.zarr")
            recompressed = compress_zarr(uncompressed, root / "recompressed.zarr")

            first = zarr.open_group(str(compressed), mode="r")
            plain = zarr.open_group(str(uncompressed), mode="r")
            last = zarr.open_group(str(recompressed), mode="r")
            self.assertEqual(first.attrs["compressor"], "zstd")
            self.assertEqual(plain.attrs["compressor"], "none")
            self.assertEqual(last.attrs["compressor"], "zstd")
            for chromosome, expected in {"chr1": "ACGTNNN", "chr2": "TTA"}.items():
                self.assertTrue(np.array_equal(first[chromosome][:], plain[chromosome][:]))
                self.assertTrue(np.array_equal(plain[chromosome][:], last[chromosome][:]))
                self.assertEqual(decode_4bit(bytes(last[chromosome][:]), last[chromosome].attrs["logical_length"]), expected)
            stats = store_statistics(recompressed)
            self.assertEqual(stats["chromosomes"], 2)
            self.assertEqual(stats["logical_bases"], 10)
            self.assertEqual(stats["packed_bytes"], 6)
            self.assertGreater(stats["apparent_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
