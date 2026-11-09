"""Import isolation for reproduction, with compatible public package exports."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PackageImportTests(unittest.TestCase):
    def run_fresh_python(self, source):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reproduction_imports_work_when_public_apis_are_unavailable(self):
        self.run_fresh_python("""
            import importlib.abc
            from pathlib import Path
            import runpy
            import sys

            blocked = {'splm_tm.index', 'splm_tm.search_encoders', 'splm_tm.alignment'}
            class BlockPublicAPIs(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in blocked:
                        raise AssertionError('Reproduction imported public API: ' + fullname)
                    return None

            sys.meta_path.insert(0, BlockPublicAPIs())
            root = Path.cwd()
            sys.path.insert(0, str(root / 'code'))
            from splm_tm import SPLMTM, Backbone, read_fasta, paths, load_align_head
            for relative in (
                'large_scale_search/frozen_search.py',
                'scope_retrieval/run_scope.py',
                'cath_s20/run_cath.py',
                'go_functional/run_go.py',
                'malidup_malisam/run_alignment.py',
            ):
                script = root / 'reproduce' / relative
                sys.path.insert(0, str(script.parent))
                runpy.run_path(str(script))
            assert not blocked.intersection(sys.modules)
        """)

    def test_public_exports_remain_available_on_demand(self):
        self.run_fresh_python("""
            import importlib
            import sys
            sys.path.insert(0, 'code')
            import splm_tm
            exports = {
                'SearchIndex': 'splm_tm.index',
                'SearchResult': 'splm_tm.index',
                'T5SearchEncoder': 'splm_tm.search_encoders',
                'LegacyESMSearchEncoder': 'splm_tm.search_encoders',
                'CandidateAligner': 'splm_tm.alignment',
                'CandidateAlignmentResult': 'splm_tm.alignment',
            }
            assert not set(exports.values()).intersection(sys.modules)
            assert set(exports).issubset(dir(splm_tm))
            for name, module in exports.items():
                value = getattr(splm_tm, name)
                assert value is getattr(importlib.import_module(module), name)
                assert value is getattr(splm_tm, name)
            namespace = {}
            exec('from splm_tm import *', namespace)
            assert set(splm_tm.__all__).issubset(namespace)
            try:
                splm_tm.nonexistent_export
            except AttributeError:
                pass
            else:
                raise AssertionError('Missing exports must raise AttributeError')
        """)


if __name__ == "__main__":
    unittest.main()
