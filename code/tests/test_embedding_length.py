"""Exercise full-length embedding and explicit truncation without model weights."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm import Backbone, SPLMTM
from splm_tm import cli, embed_cli


class LengthModel:
    """Expose the lengths reaching the model as protein vectors."""

    def __call__(self, *, batch_tokens, mode, return_embedding):
        features = batch_tokens.float().unsqueeze(1).expand(-1, 1024)
        return None, None, features, None


class EmbeddingLengthTests(unittest.TestCase):
    def setUp(self):
        self.seen = []

        def converter(batch):
            self.seen.extend(sequence for _, sequence in batch)
            return None, None, torch.tensor([len(sequence) for _, sequence in batch])

        self.backbone = Backbone(LengthModel(), converter, torch.device("cpu"),
                                 "test", Path("config.yaml"), Path("checkpoint.pth"))
        self.model = SPLMTM(self.backbone)
        self.sequence = "A" * 1200

    def test_python_embedding_preserves_full_sequence_by_default(self):
        result = self.model.embed([self.sequence], modes=("raw",), verbose=False)
        np.testing.assert_array_equal(result["emb_raw"], np.full((1, 1024), 1200))
        self.assertEqual(self.seen, [self.sequence])

    def test_python_explicit_limit_still_truncates(self):
        result = self.model.embed([self.sequence], modes=("raw",), max_length=512,
                                  verbose=False)
        self.assertEqual(self.seen, [self.sequence[:512]])
        np.testing.assert_array_equal(result["emb_raw"], np.full((1, 1024), 512))

    def test_cli_default_zero_and_explicit_limit_reach_encoder(self):
        for main, prefix in ((embed_cli.main, []), (cli.main, ["embed"])):
            for options, length in (([], 1200), (["--max-length", "0"], 1200),
                                    (["--max-length", "512"], 512)):
                with self.subTest(entry=main.__module__, options=options):
                    self.seen.clear()
                    args = prefix + ["--seq", self.sequence, "--modes", "raw", "--quiet"] + options
                    with patch.object(SPLMTM, "load", return_value=self.model), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(args), 0)
                    self.assertEqual(self.seen, [self.sequence[:length]])


if __name__ == "__main__":
    unittest.main()
