# TODO

Docs: ARCHITECTURE.md (current design, self-contained) · REPORT.md (history +
justifications) · HANDOFF.md (operational notes, read first).

## Run #4 completed (July 8) — REGRESSION, awaiting user decision
Square patches (N=4900) + FiLM + sigma_min 0.001, 60 ep from scratch:
- binding probe split −0.08 (still no binding; sign within noise of global disorder)
- ALL samples degraded: conditioned ~0.55, UNCONDITIONAL 0.59 mean |ψ6|
  (vs 0.82 for the best circular-patch unconditional model)
- The round bundled 4 changes (square crops, coord_scale/sigma_min, FiLM, larger N),
  so the cause is not attributable without ablation. Candidates: undertraining
  (60 ep vs the 150-ep lineage that reached 0.82), reduced crop diversity
  (5 overlapping crops/box, centers confined to [35,65]²), FiLM side effects on
  optimisation, schedule shift.
Options on the table (see HANDOFF.md): ablate one factor at a time; train run-4
config much longer; or pivot to sampling-time soft-psi6 guidance (binds by
construction) on top of the best unconditional recipe.

## Findings from conditioning rounds 1-3 (July 7-8)
- Run #1: fake rim ring in training maps → fixed (interior-only rasterisation).
- Run #2: raw field values nearly constant (0.95±0.05) → model ignored field;
  map generator undertrained (2.8k steps) → blank maps. Both fixed.
- Run #3: map generator now produces interior structure; particle binding still
  ~10x too weak (half-map probe split +0.03; higher CFG w degrades without adding
  selectivity). Root cause: field is redundant with visible geometry at most noise
  levels + input-only conditioning washes out over 6 residual layers.

## Next
- [ ] Judge replication test: do boundaries appear where the target map says?
      If binding weak → raise `FIELD_GUIDANCE` in sampling.py (1.5–4).
- [ ] Judge generated maps: layouts should have sparse thin boundaries
      (real patches: mean |psi6| ≥ 0.95). If maps look wrong, train map generator
      longer / larger.
- [ ] If unconditional quality still short: more EGNN layers (8) or more corrector
      steps in the sigma ≈ lattice-spacing band — NOT more epochs at current size
      (validation has plateaued).
- [ ] Physical validation: relax generated configurations in LAMMPS; compare energies
      vs real configurations.

## Later
- [ ] Full 10,050-particle boxes: retrain at box scale (new coord_scale ~40,
      sigma_min ~0.001, box-sized maps 128×128). Fits in 5.1 GB VRAM. See HANDOFF.md.
- [ ] Graph batching for training throughput (CoM ops must become per-graph).
- [ ] torch.load(weights_only=True) cleanup.

## Done
- [x] Rebuilt pipeline: coordinate normalisation, leakage-free conditioning,
      stabilised EGNN (RBF edges, radial features), kNN dynamic graphs.
- [x] Log-sigma noise schedule matched to physical length scales (the decisive fix).
- [x] Repulsion guidance (overlaps + defect cavities), CFG, Langevin correctors.
- [x] psi6-field conditioning + map generator (two-stage design).
- [x] Evaluation suite (evaluate.py: GT-style panels + Wasserstein metrics).
- [x] 4000-particle patches; crash-safe checkpoints; scriptable sampling CLI.
