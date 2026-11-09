"""Measure load/search resident memory in a fresh process without model weights."""
import ctypes
import json
from pathlib import Path
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splm_tm import Index, QueryEmbeddings
from splm_tm.index_store import RecordStore


def peak_rss():
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(key, ctypes.c_size_t) for key in
                          ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                           "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def main(directory):
    before = peak_rss()
    tracemalloc.start()
    index = Index.load(directory, check_model=False)
    after_load = peak_rss()
    query = QueryEmbeddings(["q"], np.ones((1, index.meta["dim"]), dtype=np.float32),
                            index.meta["setup"]["encoder"], index.meta["max_length"])
    result = index.search(query, topk=3, chunk_size=1024, exact=True,
                          prefilter="pfamfam", query_pfam={"q": ["PF00001"]})
    _, python_peak = tracemalloc.get_traced_memory()
    print(json.dumps(dict(count=len(index), baseline_rss=before, load_peak_rss=after_load,
                          search_peak_rss=peak_rss(), python_peak=python_peak,
                          returned=len(result.hits[0]), max_score_block=result.metadata["max_score_block"])))


if __name__ == "__main__":
    main(sys.argv[1])
