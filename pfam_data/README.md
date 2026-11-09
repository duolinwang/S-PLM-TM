# pfam_data/ — data for the Pfam prefilter

Only needed for `--prefilter pfamfam` / `--prefilter pfamclan`. **Without a prefilter
(the default), this directory is not used at all.**

## Included

| File | Size | Contents |
|---|---|---|
| `Pfam-A.clans.tsv` | 1.8 MB | **Pfam 36.0** family → clan mapping. md5 `9169b987991322123653e9e17ed789cd`, 20,795 rows, of which 8,927 families belong to a clan |

### Why Pfam 36.0 is pinned rather than `current_release`

Pfam reorganises clans between releases, and clan membership determines the
prefilter's candidate set directly. Checking the 878 distinct family→clan assertions
from the published PfamScan output against each release:

| Pfam release | Assertions disagreeing with the published run |
|---|---|
| 34.0 | 147 / 878 |
| 35.0 | 139 / 878 |
| **36.0** | **0 / 878** ← the release the published numbers used |
| 37.0 | 1 / 878 |
| 38.x (current) | 15 / 878 |

Switching to `current_release` changes roughly 2% of clan assignments, which changes
the candidate sets, which means **reproduced numbers will not match MODEL_CARD — and
nothing will warn you**. Hence the pin. Using a newer Pfam is perfectly reasonable
scientifically; it is just no longer a reproduction.

## Downloaded separately (300 MB, not shipped)

Run once:

```bash
bash code/scripts/setup_pfam.sh          # installs into this directory by default
```

It fetches the **Pfam 36.0** HMM library and PfamScan from the EBI FTP site and runs
`hmmpress`, producing the layout `pfam.py` and `search.py` look for by default:

```
pfam_data/
  Pfam-A.clans.tsv        (shipped)
  Pfam_db/
    Pfam-A.hmm            + .h3f / .h3i / .h3m / .h3p from hmmpress
    Pfam-A.hmm.dat
  PfamScan/
    pfam_scan.pl  Bio/...
```

### Prerequisites: HMMER and Perl

`pfam_scan.pl` is a Perl script that calls HMMER's `hmmscan`:

```bash
conda install -c bioconda hmmer perl-bioperl perl-moose
# or: apt-get install hmmer bioperl
```

`setup_pfam.sh` checks for both up front and tells you what is missing, rather than
failing partway through.

## Licensing

Pfam data is **CC0 1.0 (public domain)** and may be redistributed, which is why
`Pfam-A.clans.tsv` ships here. PfamScan itself is **GPL-2.0+** and is therefore
**not** bundled — `setup_pfam.sh` downloads it — so that GPL code does not end up
inside this package's PolyForm Noncommercial license (see the root `LICENSE`).

Cite Pfam as: Mistry, J. et al. Pfam: The protein families database in 2021.
*Nucleic Acids Res.* **49**, D412–D419 (2021).
