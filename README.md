# Signless Topological Signal Processing — reproduction code

Code to reproduce every experimental result in *Signless Topological Signal Processing*.

Everything is downloaded, built and computed from public sources by the scripts below.

```
code_to_share/
├── README.md
├── requirements.txt
├── src/                          # the library
│   ├── complex.py                #   simplicial complex, Q_{p,q}, L^un_{p,q}, Hodge B_p
│   ├── bands.py                  #   interaction-order filtration V_k, bands W_k, pi_k
│   ├── regularizers.py           #   cohesion, polarization, R_ord, oriented baselines
│   └── recon.py                  #   trials, the two non-Tikhonov baselines, metrics
└── scripts/
    ├── 00_download.py            # fetch the four raw datasets
    ├── 01_build_complexes.py     # -> data/processed/*.npz
    ├── 02_reconstruction.py      # the experiments -> results/recon_*.csv
    ├── 03_collate.py             # -> results/recon_headline.csv, energy_profile.csv, ...
    ├── 04_acm_null.py            # permutation null for the acm-coauth signal
    ├── 05_figure_experiments.py  # -> figures/fig_experiments.pdf        (paper Fig. 3)
    ├── 06_table_experiments.py   # -> LaTeX body of the results table    (paper Table 1)
    ├── 07_figure_bands.py        # -> figures/fig_bands_illustration.pdf (paper Fig. 2)
    ├── 08_verify.py              # re-derives every paper number from results/
    └── fig_edge_operators/
        └── make_figure.py        # -> figures/fig1_edge_operators.pdf    (paper Fig. 1)
```

## 1. Setup

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

All scripts are run from the package root and take their paths from their own location,
so no environment variable or installation step is needed.

Set the BLAS thread count to 1 before running the reconstruction: the work is already
parallel across datasets, and letting each process grab every core makes the whole thing
slower.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 2. Full pipeline

```bash
python scripts/00_download.py                        
python scripts/01_build_complexes.py                 

python scripts/02_reconstruction.py NDC-substances   
python scripts/02_reconstruction.py fp-landscape  &  
python scripts/02_reconstruction.py acm-coauth    &  
python scripts/02_reconstruction.py tags-math     &  
wait

python scripts/04_acm_null.py                        
python scripts/03_collate.py                         
python scripts/05_figure_experiments.py              
python scripts/06_table_experiments.py               

python scripts/07_figure_bands.py                    
python scripts/fig_edge_operators/make_figure.py     

python scripts/08_verify.py                          
```


## 3. Data

| dataset | source | fetched by `00_download.py` from |
|---|---|---|
| `fp-landscape` | Poelwijk, Socolich & Ranganathan, *Nat. Commun.* 10:4213 (2019), Supplementary Data | `static-content.springer.com/.../41467_2019_12130_MOESM8_ESM.xlsx` |
| `tags-math` | Benson et al., *PNAS* 2018, `tags-math-sx` | Google Drive, id linked from `cs.cornell.edu/~arb/data/tags-math-sx/` |
| `NDC-substances` | Benson et al., *PNAS* 2018, `NDC-substances` | Google Drive, id linked from `cs.cornell.edu/~arb/data/NDC-substances/` |
| `acm-coauth` | ACM bibliographic data, as redistributed by the HAN repository (Wang et al., WWW 2019) | `github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat` |

`00_download.py` skips files already present, so it is safe to re-run.

### How the complexes are built

`01_build_complexes.py` writes one `.npz` per dataset with the simplices at levels 0–3,
the signal at levels 1–3, and a replicate pair used for the split-half reliability.

For the two event datasets the complex is the downward closure of the observed
hyperedges, and the level-*k* signal counts events whose participant set is **exactly**
that *k*-simplex (`log1p`-transformed). A simplex present only as a face of some larger
observed set therefore gets a real zero, and nothing at level *k* is ever computed from
level < *k*. For `fp-landscape` every subset of the 13 mutated sites was assayed, so the
complex is the full simplex and the signal is the measured brightness. For `acm-coauth`
the structure comes from the paper × author matrix and the signal — the *k*-way TF-IDF
co-moment of the authors' term vectors — from the independent term × paper matrix.

Expected output (this is also written to `results/dataset_table.csv`):

| dataset | max_size | budget | N₀, N₁, N₂, N₃ | dim W_{p+1}, p=2 / p=3 | split-half rel. p=2 / p=3 |
|---|---|---|---|---|---|
| `fp-landscape` | — | — | 13, 78, 286, 715 | 208 / 429 | 1.00 / 1.00 |
| `tags-math` | 6 | 25 | 25, 297, 1093, 846 | 797 / 64 | 0.78 / 0.37 |
| `NDC-substances` | 6 | 55 | 51, 175, 204, 167 | 75 / 26 | 0.83 / 0.73 |
| `acm-coauth` | 8 | 800 | 542, 1064, 689, 290 | 65 / 5 | 0.75 / 0.69 |

## 4. Hyperparameters

**Dataset configuration** (the `max_size` cap and node budget in the table above) is
hard-coded in `01_build_complexes.py` at the selected values. Both follow the two rules
stated in the paper — `max_size` is the smallest cap making the p=3 top band
non-degenerate, and the budget maximises N₂/N₀ subject to N₂ ∈ [200, 1200]. 

**Hyperparameters**: 
the grids searched are fixed in `scripts/02_reconstruction.py`:

| | value |
|---|---|
| α | `logspace(-3, 3, 7)` |
| γ | `logspace(-3, 1, 3)` |
| interaction-order profile β | `k**4`, `cut_2`, `cut_3` |
| cohesion / polarization | every q < p |
| oriented Tikhonov | 6 Hodge Laplacians: {lower, upper, full} × {low-pass, high-pass} |
| secondary regularizer Γ | `I` or `G` (imputation only; denoising fixes `Γ = I`) |
| trials | 20, or 10 when N_p > 1000 (`tags-math` p=2) |
| noise / missingness | σ ∈ {0.25, 0.5, 1.0}, 30/50/70 % missing |


