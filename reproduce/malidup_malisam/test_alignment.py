"""CPU checks for the bundled alignment data and the exact reference comparison."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alignment as ra  # noqa: E402


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reference_input_has_valid_residue_coordinates(self):
        path=self.root/"pairs.csv"
        path.write_text("header\npair,x,y,AC,AD,z,z,AC-,A-D\n")
        self.assertEqual(ra.load_pairs(path),[("pair","AC","AD","AC-","A-D")])
        path.write_text("header\npair,x,y,AC,AD,z,z,AW-,A-D\n")
        with self.assertRaises(ValueError):
            ra.load_pairs(path)

    def test_bundled_pairs_match_reference_order(self):
        reference=ra.reference()
        for dataset,count in (("malidup",241),("malisam",130)):
            pairs=[row[0] for row in ra.load_pairs(ra.HERE/"data"/f"{dataset}_hec.csv")]
            self.assertEqual(len(pairs),count)
            self.assertEqual(pairs,[row["pair"] for row in reference[dataset]["pairs"]])

    def test_comparison_separates_float_noise_from_real_differences(self):
        # The exact helper stays exact; the gate is the tolerant one.
        self.assertTrue(ra.differences({"x":.5+1e-12},{"x":.5}))
        self.assertFalse(ra.tolerant_differences({"x":.5+1e-12},{"x":.5})[0])
        self.assertTrue(ra.tolerant_differences({"x":.6},{"x":.5})[0])
        self.assertTrue(ra.tolerant_differences({"mean":{"f1":.6}},{"mean":{"f1":.5}})[0])
        self.assertTrue(ra.tolerant_differences({"pair":"b"},{"pair":"a"})[0])

    def test_reference_itself_passes_and_is_reported_bit_exact(self):
        metrics=json.loads(json.dumps(ra.reference()))
        ra.finish(self.root/"same",metrics,{})
        report=json.loads((self.root/"same/comparison.json").read_text())
        self.assertEqual(report["status"],"exact_metric_match")
        self.assertTrue(report["bit_exact"])
        self.assertEqual(report["max_abs_deviation"],0.0)

    def test_last_digit_noise_passes_but_is_reported(self):
        metrics=json.loads(json.dumps(ra.reference()))
        metrics["malidup"]["pairs"][0]["f1"]+=1e-12
        metrics["malidup"]["mean"]["f1"]+=1e-12
        ra.finish(self.root/"noise",metrics,{})
        report=json.loads((self.root/"noise/comparison.json").read_text())
        self.assertEqual(report["status"],"match_within_tolerance")
        self.assertFalse(report["bit_exact"])
        self.assertGreater(report["max_abs_deviation"],0.0)
        self.assertFalse(report["differences"])

    def test_difference_that_moves_a_reported_number_fails(self):
        metrics=json.loads(json.dumps(ra.reference()))
        metrics["malidup"]["mean"]["f1"]+=0.01
        with self.assertRaises(SystemExit):
            ra.finish(self.root/"changed",metrics,{})
        report=json.loads((self.root/"changed/comparison.json").read_text())
        self.assertEqual(report["status"],"different")
        self.assertTrue(report["differences"])

    def test_single_pair_beyond_tolerance_fails(self):
        metrics=json.loads(json.dumps(ra.reference()))
        metrics["malisam"]["pairs"][0]["precision"]+=0.01
        with self.assertRaises(SystemExit):
            ra.finish(self.root/"pair",metrics,{})


if __name__=="__main__":
    unittest.main()
