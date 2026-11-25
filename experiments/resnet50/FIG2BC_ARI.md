# Fig 2B / 2C: ARI scores with Pretrained ResNet50 and ViT-S

Work is in the context of EPFL Master ML course, credited project.

This file describes how **ResNet50** (ImageNet) and **ViT-S/16** (DINO on ImageNet) were
added on figures 2B & 2C of the original paper.
Done starting from the original file `embeddings_analysis/Fig_2BC_ARI_score.py`.

- **Fig 2B, batch effect**: ARI between K-means clusters and the batch label (`experiment`, E1 to E4).
  Calculations performed on all cells, K=4. **Lower is better** (less batch effect).
- **Fig 2C, ALS:TIC**: ARI between K-means clusters and the biological condition `mut_cond` (mutation ALS/control × Basal/Low TIC/High TIC).
  Calculation done on each batch individually, K=4 or K=6 depending on the number of conditions in the batch. **Higher is better.**

Original files (`embeddings_analysis/…`, `utils/mypalette.py`) were not modified.

---

## 1. Files

| File | Function |
|---|---|
| `experiments/resnet50/fig2c_ari.py` | Drives calculations: UMAP + K-means + ARI for each model. Computes 2C ARI by default (`--target_label mut_cond --batch_mode True`). |
| `experiments/resnet50/fig2b_ari.py` | Wrapper which runs `fig2c_ari.py` with `--target_label experiment --batch_mode False` to get Fig 2B. |
| `experiments/resnet50/fig2bc_ari_plot.ipynb` | Reads yaml of the results and draws figures accordingly. Adapted from `embeddings_analysis/Fig_2BC_ARI_boxplot.ipynb`. |
| `experiments/resnet50/extract_resnet50.py` | ResNet50 embeddings extraction pipeline. |

---

## 2. Input data

### Set of cells

`data/annotations_processed_crops.csv` CSV of annotated crops, downloaded from Zenodo. It contains **19 326 cells** once SOD1 is excluded:

| Batch | Cells | Conditions |
|---|---|---|
| E1 | 6 321 | 6 |
| E2 | 4 079 | 4 |
| E3 | 3 132 | 4 |
| E4 | 5 794 | 6 |

Some columns were renamed to fit the code: `well_row → row`, `well_col → col`, `well_id → well`, `img_id → img_code`.
Original file can be found untouched under `data/annotations_processed_crops_zenodo_original.csv`.

> ⚠️ **SAME SET OF CELLS AS PAPER? ASKED ON SLACK, waiting for answer.**
> The original analysis uses `annotations_HDBSCAN_filter_no_adjacent_single.csv` (~21 900 crops, HDBSCAN filter), which we don't have.
> Meanwhile, **all models were recomputed on the same Zenodo set** so that scores are comparable with each other.
> Our values are therefore not directly comparable to the ones in the paper.

### Embeddings (`embeddings/`, ignored by git)

| Model (name in yaml) | File | Cell alignment |
|---|---|---|
| `Pretrained\nResNet50` | `resnet50_imagenet/latent_embeddings_train.pickle` (19 326 × 2048) | No image names. Rows follow Zenodo CSV order (`shuffle=False` at extraction). |
| `Pretrained\nViT-S` | `vits16_dino_imagenet/latent_embeddings_train.pickle` (19 326 × 384) | No image names. Same extraction script and row order as ResNet50 (see below). |
| `Standard\nSimCLR` | `SimCLR.csv` (`img_code` + 512 dims) | Via `img_code`. |
| `Augmented\nSimCLR` (SI-SimCLR) | `SI_SimCLR.csv` (`img_code` + 512 dims) | Via `img_code`. |
| `hand-crafted` | `handcrafted.csv` | Via `img_code`. |
| `Pretrained\nResNet18` | `pretrained_resnet_embeddings.csv` | Via `img_code`. |
| `uniDINO` | `uniDINO.csv` | Via `path`. |
| `CellPaint SSL` | `CellPaint-SSL.csv` | Via `img_code`. |

All models were aligned on the annotations CSV via `file_name`, the image name without folder nor extension.
An `assert` checks that the ordering of cells is identical for all models.

**Some remarks on the alignments**

- ViT-S was extracted with the same script as ResNet50 (`extract_resnet50.py` with `backbone: dino_vits16`, commit `b35debf`),
  so its rows follow the same order (19 326 rows, as in the Zenodo CSV).
  As an extra check, for 3 000 cells the 10 nearest neighbours in both embeddings overlap by 11 % with the same row order,
  vs 2.4 % when shifted by one row and 0.4 % when shuffled (chance).
- The original SimCLR pickles (`240517_11/`, `240813_5/`, ~21 900 rows) have no image names and their order could not be recovered.
  The authors sent `SimCLR.csv` and `SI_SimCLR.csv` with `img_code` instead.

---

## 3. Protocol (identical to original analysis)

1. **Filtering**: removing SOD1 and FOV 5 from well 52 of E1. FOV already absent from Zenodo data.
2. **Preprocessing**: `StandardScaler` on each embedding.
   - For hand-crafted, we additionally remove features with null variance.
   - For uniDINO and CellPaint SSL, we remove features of variance ≤ 1e-5.
3. **UMAP** in 2D with `random_state=42`, on a 14 parameters grid:
   - `n_neighbors` ∈ {15, 50, 100, 150, 200, 300, 600};
   - `min_dist` ∈ {0.001, 0.1}.
4. **K-means** on the UMAP projection, with `random_state=42` and `n_init=10`.
   K is the number of classes: 4 for 2B, 4 or 6 depending on the batch for 2C.
5. **ARI** between clusters and true labels. For each model (and each batch in 2C), we keep the **best ARI score** of the 14 UMAP settings.
   In case of a tie, we keep the first setting in grid order.

### Changes from the original script

Calculations are the same, only code organization changes:

- added ResNet50 & ViT-S (ImageNet);
- SimCLR & SI-SimCLR read from CSV instead of pickles;
- option `--models` to only compute certain models;
- option `--annotations` to choose a set of cells; each set gets its own results subfolder;
- option `--quick` to compute fast test on single UMAP setting;
- UMAP calculations in parallel on all cores (`--n_jobs`, joblib);
- results are **added** to existing yaml.

---

## 4. Run calculations

Run from repo root, in `si-simclr` env. Runs take up to ~2 h and must not be interrupted by the computer going to sleep:

- run inside tmux (or with `nohup`), so that closing the terminal does not stop the job;
- prevent sleep during the run:
  - macOS: prefix the command with `caffeinate -i` (as below);
  - Linux: prefix the command with `systemd-inhibit --what=idle:sleep` instead;
  - Windows: disable sleep in the power settings for the duration of the run;
  - remote server / cluster: nothing to do.

```
caffeinate -i python experiments/resnet50/fig2b_ari.py \
  --models resnet50 vits 2>&1 | tee logs_2b.txt

caffeinate -i python experiments/resnet50/fig2c_ari.py \
  --models resnet50 vits 2>&1 | tee logs_2c.txt
```

- Possible `--models`: `resnet50 vits simclr si_simclr handcrafted resnet18 unidino cellpaint`. All models are computed if not specified.

**Computation time** (Mac M2, 8 cores):

| Calculation | UMAP fits | Time |
|---|---|---|
| 2C, ResNet50 alone | 56 | 40 min |
| 2B, 7 other models | 98 | 47 min |
| 2C, 7 other models | 392 | 2 h 10 |

### Results

```
results/analysis/ARI_UMAP_batch/annotations_processed_crops/best_umap_kmeans_ari.yaml   # Fig 2B
results/analysis/ARI_UMAP/annotations_processed_crops/best_umap_kmeans_ari.yaml         # Fig 2C
```

Each yaml gives, for each model and each batch (`all` for 2B), the best ARI (`ari`), the K used (`k_used`) and the UMAP setting (`n_neighbors`, `min_dist`). With `--quick`, results go into `best_umap_kmeans_ari_quick.yaml`.

---

## 5. Figures

Can be computed via `experiments/resnet50/fig2bc_ari_plot.ipynb`, with kernel **Python (si-simclr)**.
Output is 2 PDFs:

```
results/analysis/ARI_figures/annotations_processed_crops/batch_ARI.pdf     # Fig 2B (barplot)
results/analysis/ARI_figures/annotations_processed_crops/ALS_TIC_ARI.pdf   # Fig 2C (boxplot, one dot per batch)
```

- **New models**: ViT-S and ResNet50 have their name in red.
- **Settings**: order, colors, set of cells (`ANNOTATIONS`) can be tweaked in the first cell of the notebook.

---

## 6. ARI results (Zenodo set, 19 326 cells)

**Fig 2B: ARI batch effect (lower is better)**

| Model | ARI |
|---|---|
| Hand-crafted | 0.080 |
| uniDINO | 0.106 |
| CellPaint SSL | 0.038 |
| **Pretrained ViT-S** | 0.204 |
| Pretrained ResNet18 | 0.145 |
| SimCLR | 0.097 |
| SI-SimCLR | 0.331 |
| **Pretrained ResNet50** | 0.164 |

**Fig 2C: ARI ALS:TIC per batch (higher is better)**

| Model | E1 | E2 | E3 | E4 | Mean |
|---|---|---|---|---|---|
| Hand-crafted | 0.032 | 0.087 | 0.058 | 0.033 | 0.052 |
| uniDINO | 0.050 | 0.089 | 0.048 | 0.038 | 0.056 |
| CellPaint SSL | 0.046 | 0.018 | 0.009 | 0.006 | 0.020 |
| **Pretrained ViT-S** | 0.140 | 0.128 | 0.250 | 0.066 | 0.146 |
| Pretrained ResNet18 | 0.112 | 0.125 | 0.111 | 0.064 | 0.103 |
| SimCLR | 0.064 | 0.082 | 0.065 | 0.043 | 0.064 |
| SI-SimCLR | 0.169 | 0.223 | 0.183 | 0.104 | 0.170 |
| **Pretrained ResNet50** | 0.112 | 0.112 | 0.148 | 0.063 | 0.109 |

### Observations

- **ResNet50** has similar results to ResNet18.
- **ViT-S (DINO)** is the second best model for 2C, mean close to SI-SimCLR (0.146 vs 0.170).
  Its mean is driven by E3; SI-SimCLR is higher on 3 of 4 batches.
- The two best models in 2C, **SI-SimCLR and ViT-S, also have the strongest batch effect** in 2B (0.331 and 0.204).

### Limitations

- **Optimistic scores**: the best of 14 UMAP settings is kept (same bias for all models).
- **Single random seed** for UMAP and K-means: small differences between models may not be robust.
- **Only 4 batches** in 2C: no statistical test between models is possible.

---

## 7. Local installation (not versioned)

These files are specific to our local setup and are not versioned:

- `environment_macos.yml`: conda env for macOS arm64, excluded via `.git/info/exclude`.
  It uses **umap-learn 0.5.7**, because version 0.6 requires scikit-learn ≥ 1.6 while env is on 1.5.
- `utils/mypath.py`: annotations local path, protected by `git update-index --skip-worktree`.
- `configs/env.yml`: paths of images, embeddings and results. Is in `.gitignore`.
- `embeddings/`, `results/` and CSVs of `data/`: ignored by git.
