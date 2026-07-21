# Equivariant GNN Diffusion for Grain-Boundary Patterns

Generative model for new, physically plausible configurations of a 2D bidisperse
soft-matter system — ~10⁴ particles of diameter 1.0 plus ~0.5% "defect" particles of
diameter 1.4, arranged as a polycrystal (triangular-lattice grains of varying
orientation, separated by thin grain boundaries, with cavities around the defects).

The generator is a **two-stage diffusion pipeline**:

1. **Stage 1 — map generator** (`map_generator.py`): a small image-diffusion model that
   samples a coarse 64×64 |ψ6| *map* — the mesoscale grain/boundary layout.
2. **Stage 2 — particle generator** (`egnn.py` + `diffusion.py`): an E(2)-equivariant
   graph-neural-network diffusion over particle positions, optionally conditioned on a
   ψ6 map, that realises a layout as an actual particle configuration.

For the full design rationale read `docs/ARCHITECTURE.md`; for the project history and
justification of every design choice read `docs/REPORT.md`; for operational notes,
invariants, and known issues read `docs/HANDOFF.md`.

---

## Folder layout

```
equivariant_gnn_diffusion/
├── src/          all Python (flat — the scripts import each other as siblings)
├── docs/         ARCHITECTURE.md, REPORT.md, HANDOFF.md, TODO.md
├── data/         config/ (raw LAMMPS)  npy/ (converted boxes)  patches/ (train,val)
├── checkpoints/  trained model weights (+ coord_scale/config baked into each .pt)
└── outputs/      generated_maps/  generated_samples/  results/ (analysis PNGs)
```

## How to run (important)

Always run from the **repository root**, invoking scripts by their path in `src/`:

```
python src/<script>.py [...]
```

This is required, not stylistic: Python puts the script's own directory (`src/`) on
`sys.path` so the sibling imports (`from egnn import …`) resolve, while the working
directory stays at the root so the relative `data/`, `checkpoints/`, and `outputs/`
paths resolve. Running from *inside* `src/` breaks the data paths.

Use the project's torch environment, and set the plotting backend to a non-interactive
one (the scripts call `plt.show()`):

```
MPLBACKEND=Agg python -u src/<script>.py
```

---

## Data flow

```
LAMMPS configs ──dataset.py──▶ data/npy/*.npz ──make_patches.py──▶ data/patches/{train,val}/*.npz
                                                                         │
                                        ┌────────────────────────────────┤
                                        ▼                                ▼
                              map_generator.py                     train_patched.py
                        (learns ψ6-map layouts)              (learns to denoise particles)
                                        │                                │
                                        ▼                                ▼
                              outputs/generated_maps/          checkpoints/model_*.pt
                                        │                                │
                                        └──────────────┬─────────────────┘
                                                       ▼
                                                  sampling.py
                                        (stage-2 generation, optionally
                                         conditioned on a map; CFG + repulsion)
                                                       │
                                                       ▼
                                        outputs/generated_samples/*.npz
                                                       │
                                                       ▼
                                                  evaluate.py
                                        (ψ6 / Voronoi / g(r) panels + metrics
                                         → outputs/results/*.png)
```

---

## Python file reference

### Core library modules (imported by the scripts; not run directly)

| File | Purpose |
|------|---------|
| `egnn.py` | The stage-2 denoiser. `EGNNUnconditionalDenoiser` (E(2)-equivariant GNN) with per-layer FiLM conditioning, radial features for global-extent awareness, RBF edge-distance features, and a sinusoidal time embedding. `EGNNLayer` is one message-passing block; `mlp` and `SinusoidalTimeEmbedding` are helpers. |
| `diffusion.py` | `PositionDiffusion` — the DDPM process over positions: the log-σ noise schedule (`logsigma_beta_schedule`), forward noising (`q_sample`), the training loss, and the reverse sampler with classifier-free guidance, soft-disk repulsion guidance, radial x̂₀ thresholding, and optional Langevin correctors. Also `remove_com` and `build_knn_edges`. |
| `field_conditioning.py` | Everything for ψ6-map conditioning: build a coarse map from a clean patch (`make_psi6_map`, interior-only to avoid rim artifacts), `FieldSampler` (bilinear lookup of the map at a particle's current position, contrast-normalised, 2-channel), and `NullField` (the unconditional token). |
| `node_features.py` | `build_node_scalars` — freud-based per-particle structural descriptors (Voronoi area/perimeter, coordination, neighbour distances, ψ6, ψ4, size, type) and `normalise_scalars`. **Used for evaluation only**, never as model conditioning (they leak the answer). |
| `g_r_analysis.py` | Radial distribution function g(r) utilities (compute, average over configs, estimate the first-minimum cutoff, plot). Library used by `dataset.py`. |

### Data preparation

| File | Purpose |
|------|---------|
| `dataset.py` | LAMMPS reader + g(r) driver. `--convert` re-reads `data/config/*` and writes `data/npy/*.npz` (with box_size). Without a flag, computes and plots g(r) over a few boxes. |
| `make_patches.py` | Cuts the full boxes into fixed-size **square** training patches (`N_TARGET`≈4900 particles, several random full-square crops per box) and writes them to `data/patches/{train,val}/`. Re-run whenever the patch size changes; it clears stale patches first. |

### Training

| File | Purpose |
|------|---------|
| `train_patched.py` | **Main training loop** for the stage-2 particle model. Loads patches, computes the global `coord_scale`, trains with AdamW + gradient clipping + EMA, supports ψ6-field conditioning (with classifier-free dropout) and resume-from-checkpoint, and saves `model_best/last/final.pt` (each stores weights + EMA + coord_scale + config). Key knobs at the top: `RESUME_FROM`, `FIELD_COND`, `SCHEDULE`, `N_EPOCHS`. Also defines `identity_scalars` (the size/type encoding) used everywhere. |
| `map_generator.py` | **Stage-1 model.** `train` fits a small UNet image-diffusion to the ψ6 maps of all training patches; `sample` draws new layout maps to `outputs/generated_maps/`. Run as `python src/map_generator.py train` / `... sample --n 4`. |
| `overfit_test.py` | Sanity check: overfit the denoiser to a single patch (or a synthetic lattice) and sample from noise. Isolates model/training/sampling correctness from dataset-scale effects. Writes `outputs/results/overfit_comparison.png`. |
| `train_on_boxes.py` | **DEPRECATED.** The original full-box trainer against an earlier API; kept for reference only, does not run against the current modules. |

### Generation

| File | Purpose |
|------|---------|
| `sampling.py` | **Stage-2 generation driver.** Loads a checkpoint (architecture/schedule/coord_scale read from it) and generates configurations to `outputs/generated_samples/`. Three conditioning modes via constants: `TARGET_MAP` (a generated `.npy` map → two-stage), `TARGET_PATCH` (replicate a real patch's map + composition), or neither (unconditional). CLI: `--map <path.npy>`, `--prefix <name>`. |
| `generate_best.py` | Convenience script to generate N samples from the best known-good **unconditional** checkpoint (`model_uncond_logsigma_backup.pt`) into a subfolder, with an overview figure. `python src/generate_best.py --n-samples 5`. |

### Evaluation

| File | Purpose |
|------|---------|
| `evaluate.py` | Structural evaluation of any configuration (generated or real). Renders the four ground-truth-style panels — density, \|ψ6\|, arg(ψ6), Voronoi sides — and, with `--ref`, prints Wasserstein distances between the generated and reference distributions of \|ψ6\|, nearest-neighbour distance, Voronoi area, and coordination. Delaunay-based, open-boundary aware. |
| `probe_binding.py` | Acceptance test for map conditioning: conditions on an extreme half-disorder / half-crystal map and measures the \|ψ6\| difference between the two halves. A large split means the model follows the map; a near-zero split means it ignores it. |

---

## Typical workflows

All commands are run from the repo root; prefix with `MPLBACKEND=Agg` and use the torch
environment's python.

**One-time data prep**
```
python src/dataset.py --convert           # LAMMPS configs → data/npy/*.npz
python src/make_patches.py                # data/npy → data/patches/{train,val}
```

**Train the particle model**
```
python src/train_patched.py               # writes checkpoints/model_{best,last,final}.pt
```

**Unconditional generation + evaluation**
```
python src/generate_best.py --n-samples 5
python src/evaluate.py outputs/generated_samples/best_model/best_000.npz
```

**Two-stage (new pattern from nothing)**
```
python src/map_generator.py train
python src/map_generator.py sample --n 4
python src/sampling.py --map outputs/generated_maps/map_000.npy --prefix twostage
python src/evaluate.py outputs/generated_samples/twostage_000.npz --ref data/patches/val/<patch>.npz
```

**Check whether conditioning binds**
```
python src/probe_binding.py --w 1.0 2.0
```

---

## Key conventions & gotchas

- **Coordinates are normalised** by a per-checkpoint `coord_scale` (stored in the `.pt`);
  the DDPM prior is unit-variance. Never mix a checkpoint with a different scale.
- **Type column is inverted**: in this dataset `type 0 = large defect (1.4)`,
  `type 1 = small (1.0)`. Always derive defect identity from `size > 1.2`.
- **Structural descriptors are for evaluation only** — feeding per-particle ψ6/Voronoi
  as conditioning leaks the answer and is unavailable at sampling time.
- The noise schedule is **log-spaced in length scale** (`logsigma`); this was the
  decisive fix that let the reverse process form and preserve crystalline order.

See `docs/HANDOFF.md` for the full list of invariants and the current state of the work.
