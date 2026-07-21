# Project Report: Equivariant GNN Diffusion for Grain Boundary Pattern Generation

**Date:** July 2026
**Goal:** Generate new, feasible grain-boundary patterns for a 2D bidisperse soft-matter
system (418 LAMMPS configurations; 10,050 particles each: 10,000 small at diameter 1.0 +
50 large "defects" at diameter 1.4), matching the structural statistics visible in the
ground-truth analyses (ψ6 maps, Voronoi statistics, g(r)).

---

## 1. Starting point and its failure modes

The original pipeline (EGNN denoiser + DDPM over positions, trained on ~1,000-particle
circular patches) produced divergent sampling (infinite values) and unusable output even
when overfitting a single patch. Diagnosis identified four independent defects, two of
them individually fatal.

## 2. Changes made, in order, with justification

### 2.1 Coordinate normalisation (fatal bug #1)

**Change:** All positions are divided by a global `COORD_SCALE` (RMS coordinate of the
CoM-free training patches, ≈17.4 for 4,000-particle patches) before entering the
diffusion; samples are multiplied back afterwards. The scale is stored in the checkpoint.

**Justification:** DDPM's prior is a *unit* Gaussian. Raw patch coordinates had std ≈ 9–18.
The reverse process cannot bridge an order-of-magnitude scale gap: samples either
collapsed into filament-like artifacts or diverged. Squared distances up to ~10³ fed to
the edge MLP also caused the training loss spikes documented in the original code.

### 2.2 Removal of leaking conditioning features (fatal bug #2)

**Change:** The model is conditioned only on per-particle *identity* (size, type). The
nine structural descriptors (Voronoi area/perimeter, coordination, neighbour distances,
ψ6, ψ4) were removed from the model input and are now used exclusively for evaluation
(`evaluate.py`), and — in coarse field form — for controllable conditioning (§2.9).

**Justification:** Seven of the nine features are computed *from the clean positions*:
during training they leak the answer, so the model learns to rely on them; at sampling
time they do not exist (they were zeroed), presenting the network with an input
distribution it never saw. Classic train/test mismatch.

### 2.3 Stabilised EGNN layer

**Changes** (following Satorras et al. EGNN as modified in Hoogeboom et al.'s E(3)
diffusion work):
- coordinate messages direction-normalised: `rel/(dist+1) · φ(m_ij)` — bounded updates;
- final linear layer of every coordinate MLP initialised near zero — stacked layers
  cannot amplify each other before training finds the right scale;
- sigmoid attention on messages — kNN graphs at high noise contain irrelevant
  neighbours the layer must be able to ignore;
- 24 Gaussian RBF features of edge length (max 0.4 normalised) — raw `dist`/`dist²`
  scalars are too weak a signal for an MLP to discriminate the fine distances that
  matter (nearest-neighbour spacing is ~0.06 normalised units);
- predicted noise projected into the CoM-free subspace — the target noise is CoM-free,
  so the prediction must be too;
- proper transformer-style sinusoidal time embedding — the original embedding's
  arguments spanned ≈[0,1], making different timesteps nearly indistinguishable.

The old output head `gate(h)·(x_out−x_in)` (unbounded gate × unbounded displacement) was
replaced by the plain equivariant readout `x_out − x_in`.

### 2.4 Radial awareness (translation equivariance deliberately broken)

**Change:** Each layer receives `|x_i|` as an invariant scalar and can apply a gated
coordinate update along `x_i/|x_i|`.

**Justification:** A purely local, translation-equivariant kNN model cannot know where a
particle is relative to the patch — every interior particle looks identical — so it can
never learn the patch's global extent; samples drifted into unbounded clumps. Since the
entire pipeline operates in the CoM-centred frame (data and every reverse step are
re-centred), the origin is physically meaningful and radial features are legitimate.
Rotation equivariance is retained. This fix made the global disk envelope snap into
place immediately.

### 2.5 kNN graphs, dynamic reconstruction

**Change:** Graphs are k=12 nearest-neighbour (cKDTree), rebuilt from the current noisy
positions at every training and sampling step. The earlier radius-cutoff construction
(and its memory blow-ups) is gone.

### 2.6 Sampling hygiene

**Changes:** CoM-free noise everywhere; re-centred after every reverse step; norm-based
(radial) static thresholding of the implied clean sample `x0_hat` (never clamping the
predicted noise — clamping ε breaks the reverse-process math and pins samples to the
clamp boundary, which was observed directly); EMA weights (decay 0.999) used for
sampling; optional Langevin corrector steps (predictor–corrector à la Song et al.).

### 2.7 Noise schedule matched to the data's length scales (the decisive fix)

**Change:** Replaced the cosine schedule with a log-spaced sigma ladder
(`logsigma_beta_schedule`): the effective noise-to-signal ratio is geometric from 0.002
to 60 over 1,000 steps.

**Justification:** Diagnostics (loss-vs-t profile, and a diffuse–denoise test that
noises a *real* crystal to timestep t and denoises back) showed that under the cosine
schedule the reverse process **destroyed a perfect crystal noised to t=25**
(|ψ6| 0.97 → 0.41): the schedule's minimum noise was ~8× larger than the lattice
thermal jitter, and ~90% of its steps sat at noise coarser than 3 lattice spacings.
Image-derived schedules allocate steps by absolute noise power; point patterns need
every *length scale* (field → grain → lattice → jitter) covered. Log-spacing does
exactly that. Effect: generated |ψ6| 0.42 → 0.73 in one retrain, → 0.82 after a second
round; median NN distance 0.84 → 1.00 (real: 1.05). Neither low-t oversampling nor
Langevin correctors helped while the schedule was wrong — the schedule was the
bottleneck.

### 2.8 Repulsion guidance at sampling time

**Change:** During the last 25% of reverse steps, a soft-disk overlap force (using each
particle's actual radius, so small–small contact = 1.0 and small–defect contact = 1.2)
is applied to the implied clean positions.

**Justification:** MSE-trained diffusion learns spacing statistically and leaves a tail
of overlapping pairs; hard-core exclusion is a fact we know exactly and need not learn.
The size-aware force also carves the characteristic cavities around the 1.4-diameter
defects. Sampling-only; no retraining.

### 2.9 Coarse ψ6-field conditioning + classifier-free guidance

**Change:** The model can be conditioned on a smooth 64×64 |ψ6| map (computed from the
clean patch during training via Delaunay ψ6 + Gaussian rasterisation; supplied by the
user or by the stage-1 generator at sampling time). The field is bilinearly interpolated
at each particle's *current* position at every denoising step. Trained with 15%
conditioning dropout; sampled with classifier-free guidance (weight ~2).

**Justification:** This is the correct way to "use the descriptors during generation":
unlike exact per-particle features, a coarse field is specifiable at sampling time, so
there is no train/test mismatch. It converts the hard problem (invent a global grain
layout with a local model) into an easy one (realise a given layout locally), and gives
controllable generation.

### 2.10 Two-stage generation (map generator)

**Change:** A small image-diffusion model (`map_generator.py`, ~1M-param UNet over 64×64
maps) learns the distribution of ψ6 layouts from the 1,508 training-patch maps. New
patterns are generated by sampling a layout from stage 1 and realising it with the
field-conditioned particle model.

**Justification:** Mesoscale layout statistics (grain size distribution, boundary
topology) are global properties, easy for a small image model that sees the whole map,
and structurally hard for a local particle GNN. Division of labour by length scale.

### 2.11 Data/infrastructure fixes

- **Patch size 1,000 → 4,000 particles** (~110 diameters wide): a patch must contain
  several grains and boundary junctions for the model to learn boundary statistics.
  Measured VRAM: 2.1 GB/step at N=4,000 (RTX 4050, 6.4 GB) — the original OOM was an
  artifact of the old graph construction, not a fundamental limit (N=10,050 fits at
  5.1 GB).
- **Type-column convention discovered to be inverted**: in this dataset type 0 = large
  defect, type 1 = small. All plotting/statistics now derive defect identity from size
  (>1.2); sampling-time conditioning reproduces the dataset convention.
- **Evaluation suite** (`evaluate.py`): density / |ψ6| / arg(ψ6) / Voronoi-sides panels
  (Delaunay-based, open-boundary aware, hull margin excluded) + Wasserstein distances of
  |ψ6|, NN-distance, Voronoi-area and coordination distributions vs a reference.
- Generated coordinates saved (`generated_samples/*.npz`) for evaluation and downstream
  use (e.g. LAMMPS relaxation).

## 3. Quantitative trajectory

Interior-particle statistics vs the real reference (mean |ψ6| 0.97, median NN 1.05):

| model state                              | mean \|ψ6\| | med. NN | W(\|ψ6\|) |
|------------------------------------------|------------|---------|-----------|
| original pipeline                        | — (diverged / collapsed) | — | — |
| rebuilt, cosine schedule (100 epochs)    | 0.42       | 0.84    | 0.55      |
| log-sigma schedule (60 epochs)           | 0.73       | 0.97    | 0.24      |
| log-sigma, 120 epochs                    | 0.82       | 1.00    | 0.15      |
| + field-cond. + repulsion (60 epochs)    | 0.76       | 0.99    | 0.21      |

The unconditional 120-epoch model produces clear polycrystalline structure: orientation
domains in arg(ψ6), connected boundary networks in |ψ6|, correct density and
composition. Remaining gap: grains are smaller / boundaries denser than real (real
patches have mean |ψ6| ≥ 0.95 — boundaries are sparse); this is the motivation for the
two-stage design.

## 4. Honest assessment

- Global structure, density, composition, spacing: **solved**.
- Local crystalline order: **substantially recovered** (0.42 → 0.82 across schedule and
  training rounds); validation loss has since plateaued at the current model size
  (hidden 128, 6 layers), so further gains should come from capacity or sampling-side
  changes, not more epochs.
- Grain-scale layout statistics: **the open question**; two-stage generation is the
  designed answer and is implemented but not yet validated end-to-end.
- Generated configurations should be validated (and optionally relaxed) in LAMMPS before
  scientific use: a generative model reproduces statistics, not energy minima.
