# Architecture Reference

A self-contained description of the **current** generation pipeline. No prior knowledge
of this project is assumed; where a design choice is non-obvious, the reasoning is given
in place. (For the *history* of how we got here — including the failure modes that
motivated each choice — see REPORT.md.)

---

## 1. The problem

We want to generate new, physically plausible configurations of a 2D bidisperse particle
system: ~10⁴ particles of diameter 1.0 plus ~0.5% "defect" particles of diameter 1.4,
arranged in a polycrystal — mostly triangular lattice, broken into grains of different
orientations separated by thin grain boundaries, with characteristic cavities around the
large defects. The training data is 418 simulation boxes (10,050 particles each). For
memory and statistical reasons we train on ~4,900-particle SQUARE patches (70×70
particle diameters ≈ 2–3 grain diameters, so several grains and boundary junctions per
patch; 5 random fully-inside crops per box). Square patches also match the conditioning
map's square frame exactly — earlier circular patches left the map's corners as
meaningless fill values.

A configuration is just a set of 2D points plus per-particle identity (size, type).
Generation must produce the *positions*; the identities are chosen by the user
(composition is an input, not an output).

The output we care about is judged by structural statistics: the bond-orientational
order parameter ψ6 (≈1 on a perfect triangular lattice, dips at boundaries), Voronoi
statistics, nearest-neighbour distances, and the visual grain/boundary layout.

## 2. The two-stage design

Grain-boundary patterns have structure at two very different length scales:

- **Mesoscale** (tens of particle diameters): the grain layout — how big grains are,
  where boundaries run, how they meet at junctions.
- **Microscale** (one particle diameter): lattice order, spacing, defect cavities,
  the 5/7-coordination pairs that decorate boundaries.

A particle-level graph network is inherently *local* (each particle sees ~12 neighbours,
a few hops deep), which makes it excellent at the microscale and structurally bad at
inventing mesoscale layouts. So the pipeline splits the job:

1. **Stage 1 — map generator** (`map_generator.py`): a small *image* diffusion model
   over 64×64 coarse |ψ6| maps. An image model sees the whole layout at once, so
   mesoscale statistics are easy for it. Output: a new plausible grain layout, as a
   small grayscale image.
2. **Stage 2 — particle generator** (`egnn.py`, `diffusion.py`): an equivariant GNN
   diffusion over particle positions, *conditioned* on a ψ6 map. It realises the given
   layout as an actual particle configuration with correct local physics.

Generating a brand-new pattern from nothing = sample a map from stage 1, feed it to
stage 2. The same stage-2 model also accepts maps from real configurations (controlled
regeneration) or hand-designed maps, and can run unconditionally (see §7).

---

## 3. Diffusion preliminaries (as used here)

Both stages are DDPMs (denoising diffusion probabilistic models). Briefly:

- **Forward process**: data x₀ is progressively corrupted over T=1000 steps,
  x_t = √(ᾱ_t)·x₀ + √(1−ᾱ_t)·ε, with ε ~ N(0, I). ᾱ_t (alpha-bar) decreases from ~1
  to ~0; at t=T the sample is indistinguishable from pure Gaussian noise.
- **Model**: a network is trained to predict the noise ε from (x_t, t) — plus any
  conditioning — with a simple MSE loss.
- **Reverse process (sampling)**: start from pure noise x_T ~ N(0, I) and iteratively
  denoise. At each step we form the *implied clean sample*
  x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √(ᾱ_t), then take a step toward it using the standard
  DDPM posterior mean, plus fresh noise of the appropriate (shrinking) magnitude.

Two consequences shape everything below:

**(a) The prior is a UNIT Gaussian**, so the data must be brought to roughly unit
variance. All positions are divided by a global constant `coord_scale` (the RMS
coordinate of the CoM-centred training patches; currently 17.43) on the way in and
multiplied back on the way out. Every length quoted "in normalised units" below means
"physical length / coord_scale". Key normalised lengths: patch radius ≈ 2.0,
lattice spacing ≈ 0.057, lattice thermal jitter ≈ 0.003.

**(b) Translation symmetry must be quotiented out.** The denoiser is
translation-invariant in its *inputs* (it uses only relative positions and, see §5.4,
positions in the centre-of-mass frame), which means the data distribution is only
defined up to translation. A Gaussian prior is not translation-invariant, so we work in
the **CoM-free subspace** throughout: training data is CoM-centred, the injected noise
is CoM-projected (ε ← ε − mean(ε)), every reverse step re-centres x_t, and the model's
predicted ε̂ is CoM-projected before the loss/step.

### 3.1 The noise schedule: log-spaced in length scale

This is the single most important design element for point patterns.

The schedule fixes, for each timestep t, the effective noise-to-signal ratio
σ_eff(t) = √(1−ᾱ_t)/√(ᾱ_t): how much each particle has been displaced relative to the
pattern. σ_eff is a *length scale*: at σ_eff ≈ 2 (normalised) the whole field is
scrambled; at σ_eff ≈ 0.057 particles are displaced by one lattice spacing (grain
structure survives, lattice identity is ambiguous); at σ_eff ≈ 0.003 only the thermal
jitter is affected.

Schedules designed for images (linear, cosine) allocate steps roughly uniformly in
noise *power*. For this data that puts ~90% of the steps at σ_eff coarser than 3
lattice spacings — where nothing structurally interesting happens — and, worse, the
smallest σ_eff they ever reach is ~8× larger than the lattice jitter. The consequence
(verified experimentally, see REPORT §2.7): the reverse process cannot form or even
*preserve* crystalline order; a real crystal noised by only 25 such steps and denoised
came back as a liquid.

The fix is `logsigma_beta_schedule`: σ_eff is **geometric (log-spaced) from 0.002 to
60** over the 1000 steps, i.e. ᾱ_t = 1/(1+σ_eff(t)²). Log-spacing gives every decade of
length scale — field → grain → lattice → jitter — a proportional share of the steps.
(This is the insight of Karras et al.'s EDM, expressed as a variance-preserving DDPM
schedule.) If the coordinate normalisation ever changes (different patch size, full
boxes), σ_min must be rescaled to stay below the normalised jitter.

---

## 4. Stage-2 inputs: what the particle denoiser sees

Per denoising step, the model receives:

1. **Noisy positions** x_t: (N, 2), normalised units, CoM-centred.
2. **A graph**: k=12 nearest neighbours (cKDTree), **rebuilt from the current noisy
   positions at every step** — there is no fixed graph, because at high noise the
   "true" neighbours are meaningless and at low noise the graph must reflect the
   crystallising structure.
3. **Noise level** t/T ∈ [0,1], one scalar per graph.
4. **Per-particle identity scalars** (2): size encoded as (size − 1.2)/0.2 (small → −1,
   large → +1) and type encoded as 2·type − 1. These use fixed, data-independent
   encodings so sampling never needs training-set statistics. *Caution:* in this
   dataset the type column is inverted (type 0 = large defect, type 1 = small); the
   encoding just passes through whatever convention the training data used, and
   sampling must reproduce it.
5. **Field features** (2, optional): the conditioning ψ6 map, bilinearly interpolated
   **at each particle's current position x_t**. The map lives in space, particles
   move, so the lookup is redone every step — the conditioning stays glued to space,
   not to particles. During training the map is computed from the clean patch
   (Delaunay ψ6 → Gaussian rasterisation onto a 64×64 grid, smoothed over ~2 lattice
   spacings). The lookup is encoded as two channels: a contrast-normalised value
   (F − 0.95)/0.05 — raw |ψ6| values sit at 0.95±0.05, nearly constant next to the ±1
   identity channels, and an unnormalised field is simply ignored by the network
   (observed: guidance had no effect) — and a has_field flag (1 = real map,
   0 = the unconditional null token; see §7).
   Maps are built from *interior* particles only (2-diameter hull margin excluded):
   rim particles have artificially low Delaunay ψ6 (neighbours missing outside the
   patch), and including them paints a fake dark ring on every map that both stages
   then learn instead of the real interior boundaries.

Items 4–5 are concatenated into `node_scalars` (width 4 for the field-conditioned
model, 2 otherwise; this width is baked into the checkpoint).

**What the model deliberately does NOT see:** exact per-particle structural descriptors
(Voronoi areas, coordination, per-particle ψ6, …). During training those are computed
from the clean answer — feeding them would let the model cheat — and at sampling time
they don't exist yet. The *coarse field* avoids this trap precisely because it is
something the user (or stage 1) can supply at sampling time, so training and sampling
see the same kind of input.

---

## 5. Stage-2 network: the equivariant GNN denoiser

`EGNNUnconditionalDenoiser` in `egnn.py`. ~1.3M parameters at the current size
(hidden_dim=128, n_layers=6).

### 5.1 Input embedding

- **Time embedding**: standard transformer/DDPM sinusoidal embedding of t/T (internally
  rescaled ×1000 so the frequency ladder 10000^(−k/half) resolves nearby timesteps),
  followed by a 2-layer MLP → (N, 128).
- **Scalar embedding**: a linear layer maps `node_scalars` → (N, 128).
- The two are concatenated and passed through an MLP to give the initial invariant
  node features h⁰ ∈ (N, 128). Note there are *no* meaningful per-particle features
  beyond identity + field + time: all geometric information enters through the layers.

### 5.2 One EGNN layer

Each of the 6 layers updates both the invariant features h and the coordinates x.
For every directed edge (src → dst) of the kNN graph:

1. **Geometry scalars**: rel = x_src − x_dst, dist² and dist, plus a **24-Gaussian RBF
   expansion of dist** with centres linspace(0, 0.4) in normalised units. The RBFs
   matter: an MLP reading a raw scalar distance can barely discriminate 0.05 from 0.06,
   but those are exactly the distinctions (fractions of a lattice spacing) that
   crystalline order lives on; the RBF ladder makes them linearly separable.
2. **Message**: m_ij = MLP(h_src, h_dst, dist, dist², RBF). Then **sigmoid attention**
   m_ij ← m_ij · σ(w·m_ij): a learned per-message gate. Needed because at high noise
   the kNN graph connects particles that have nothing to do with each other; the layer
   must be able to mute them.
3. **Coordinate message**: x_msg = (rel / (dist + 1)) · MLP₁(m_ij). Two safety
   properties: the direction vector rel/(dist+1) is bounded (never explodes when noisy
   points are far apart), and the final linear layer of the gating MLP is initialised
   near zero (gain 1e-3), so at the start of training every layer moves points only
   slightly — stacked layers otherwise amplify each other into the loss spikes that
   plagued the original implementation.
4. **Aggregation**: messages and coordinate messages are mean-aggregated over incoming
   edges (sum / degree — degree normalisation stops high-degree nodes from moving more).
5. **Node update**: h ← LayerNorm(h + MLP(h, agg_m, |x|)) — residual + LayerNorm.
6. **FiLM conditioning**: the per-node conditioning vector (time embedding ‖ scalar
   embedding, which includes the field channels) produces, through a zero-initialised
   linear layer *per EGNN layer*, a scale (1+Δγ) and shift β applied to the updated
   node features. Zero-init = exact identity at start of training. Rationale:
   conditioning that enters only the input embedding is washed out by the stacked
   residual+LayerNorm blocks — measured directly (guidance had almost no effect,
   half-map probe split +0.03 pre-FiLM). FiLM gives the conditioning an unmissable
   pathway into every layer.
7. **Radial coordinate update**: x ← x + agg_x + (x/(|x|+1)) · MLP₂(h, |x|), also
   near-zero initialised.

### 5.3 Output head

After the last layer, the predicted noise is simply the total coordinate displacement:
ε̂ = x_final − x_input, followed by CoM projection (ε̂ ← ε̂ − mean ε̂), because the
target noise is CoM-free by construction. There is no learned gate on the output — an
earlier gated head was an instability source.

### 5.4 Symmetry properties (read this before "fixing" anything)

- **Rotation-equivariant** (and reflection-equivariant): all invariant inputs (dist,
  |x|, field values under a co-rotating map) are unchanged by rotations; all coordinate
  updates are along equivariant directions (rel, x). Rotating the input rotates the
  output. Consequence: generated grain orientations are automatically uniformly
  distributed — no rotation augmentation needed.
- **Deliberately NOT translation-equivariant**: the radial terms |x_i| and x/|x| use
  the origin. This is valid *only because* the whole pipeline operates in the CoM
  frame, where the origin is physically meaningful (the patch centre). It is also
  *necessary*: a purely local, translation-equivariant model has no way to know where a
  particle is relative to the patch, so it cannot represent the patch's global extent —
  without radial features, samples collapse into unbounded drifting clumps (observed).
  If the pipeline is ever moved to periodic boxes, the radial machinery must be
  rethought (a torus has no centre).
- The conditioning map technically breaks rotation equivariance too (it is a fixed
  spatial function), but since training pairs (pattern, map) co-rotate, the model
  remains equivariant with respect to *joint* rotations of pattern and map — the same
  status as image-conditioned image diffusion.

---

## 6. Stage-2 training

Per gradient step (one patch = one graph; no batching across patches):

1. Load patch → positions/coord_scale, identity scalars, precomputed ψ6 map.
2. Draw one timestep t uniformly from [0, 1000) (a single t per graph — all particles
   share it).
3. Forward-noise: x_t = √ᾱ·x₀ + √(1−ᾱ)·ε with CoM-free ε.
4. Build the kNN graph on x_t; interpolate the field at x_t.
5. **Conditioning dropout**: with probability 15%, replace the field with the null
   token ([0, has_field=0]). Two purposes: (i) the same network learns a genuine
   unconditional mode, and (ii) classifier-free guidance at sampling (§7) requires a
   trained unconditional branch — the guidance direction is the *difference* between
   conditional and unconditional predictions, which only exists if both are meaningful.
   15% is the conventional range (10–20%); it is not a sensitive knob.
6. Loss = MSE(ε̂, ε). Backprop, grad-clip at 1.0, AdamW (lr 2e-4 fresh / 1e-4 resumed,
   weight decay 1e-4).
7. **EMA**: an exponential moving average of the weights (decay 0.999) is maintained
   and used for sampling — EMA weights sample noticeably cleaner than raw weights.

Checkpoints: `model_last.pt` every epoch (crash safety), `model_best.pt` on validation
improvement. A checkpoint stores model + EMA weights + coord_scale + full config
(hidden_dim, n_layers, n_steps, k, n_scalar_feats, schedule, field_conditioned), so
sampling never needs manually synchronised constants.

**Loss interpretation** (important for not chasing ghosts): predicting ε from nearly
pure noise is trivially easy at the highest t (loss → 0) and genuinely ambiguous in
the mid band where σ_eff ≈ lattice spacing — a particle displaced by one spacing could
belong to several lattice sites, so even a perfect model has high MSE there. A healthy
loss-vs-t profile is low at both ends with a hump in the middle; the *uniform-t average*
(~0.35 currently) mixes all of these and is only comparable within a fixed schedule.

## 7. Stage-2 sampling

`PositionDiffusion.sample`, driven by `sampling.py`. Start: x_T = CoM-projected unit
Gaussian. Per step:

1. Build kNN graph on current x_t; evaluate the model.
2. **Classifier-free guidance** (if a field is provided and weight w ≠ 1):
   ε̂ = ε̂_null + w·(ε̂_cond − ε̂_null). The difference isolates exactly the component
   of the prediction *caused by the map* (shared physics cancels), and w>1 extrapolates
   further in that direction — the same mechanism that makes text-to-image models obey
   prompts. Costs a second model evaluation per step. Current default w=2.
3. Form x̂₀ and apply **radial static thresholding**: rescale any particle whose
   implied clean position lies further than 4.0 (normalised) from the origin. This caps
   how far one bad prediction can throw the trajectory. It is norm-based rather than
   per-coordinate on purpose: per-coordinate clamping injects square geometry into the
   samples (observed as clumps pinned in box corners). *Never* clamp ε̂ itself — that
   silently breaks the reverse-process mean.
4. **Repulsion guidance** (last 25% of steps): add a soft-disk force to x̂₀. For each
   kNN pair closer than the sum of their contact radii (radius = diameter/2 /
   coord_scale, so small–small contact = 1.0, small–defect = 1.2 in physical units),
   push apart along the pair axis proportionally to the overlap. Justification:
   hard-core exclusion is a fact we know exactly — there is no reason to demand the
   network learn it statistically; and the size-aware force is what carves the correct
   cavities around defects. Strength 0.5, sampling-only.
5. DDPM posterior step toward x̂₀, add the scheduled noise (CoM-projected), re-centre.
6. Optional **Langevin corrector** (1 step, SNR rule 0.15): an extra
   fixed-noise-level relaxation using the score −ε̂/σ. Lets the configuration anneal
   *within* a noise level before the noise is reduced — physically, how such systems
   order. Doubles model evaluations.

Conditioning source at sampling (constants in `sampling.py`, overridable via CLI:
`--map <path.npy>` and `--prefix <output name>`):
- `TARGET_MAP` (a .npy from `map_generator.py`) → two-stage generation, brand-new
  layout; extent comes from the map-generator checkpoint's stored mean patch
  half-width.
- else `TARGET_PATCH` (a real patch) → replicate that patch's map and composition.
- else → unconditional: the null token is fed as the field.

Output: positions × coord_scale (physical units), saved with sizes/types to
`outputs/generated_samples/sample_XXX.npz`.

## 8. Stage 1: the map generator

`map_generator.py`. A vanilla image DDPM, small:

- **Data**: for each training patch, per-particle |ψ6| from a Delaunay triangulation,
  Gaussian-rasterised onto a 64×64 grid over the patch bounding square (smoothing ≈ 2
  lattice spacings); 1,508 maps, z-scored with dataset mean/std (values concentrate
  near 0.95, so raw [0,1] scaling would waste the dynamic range).
- **Model**: a compact UNet (64→32→16→8 resolution, channels 32/64/128, GroupNorm +
  SiLU, additive time embedding per block; ~1M params).
- **Diffusion**: cosine schedule, 400 steps, ε-prediction, x̂₀ clamped to ±3 (z-scored
  units) during sampling. The image-standard cosine schedule is fine *here* because a
  64×64 map has no sub-pixel length-scale hierarchy — the log-sigma argument of §3.1 is
  about point coordinates, not pixels.
- **Output**: maps clipped to [0,1], saved as .npy plus a preview PNG. The checkpoint
  stores the mean patch half-width so stage 2 knows the physical size a generated map
  corresponds to.

## 9. Evaluation

`evaluate.py` — works on any configuration (generated npz or real patch npz):

- Delaunay-based per-particle |ψ6|, arg(ψ6), coordination (≈ Voronoi sides), Voronoi
  cell areas, NN distances. All computed open-boundary (no periodic box needed);
  particles within 2 diameters of the convex hull are excluded from statistics to kill
  edge artifacts.
- Renders the four ground-truth-style panels: density, |ψ6|, arg(ψ6), Voronoi sides.
- With `--ref`, adds a second row and prints 1D Wasserstein distances between the
  |ψ6| / NN-distance / Voronoi-area / coordination distributions.
- Defect identity is always `size > 1.2` (see the type-inversion warning in §4).

Do not judge samples from the marker-scatter overview image
(`outputs/results/*s_overview.png` — markers overlap visually at N=4000 regardless of
true positions); use the evaluate panels and numbers.

## 10. Appendix: narrative walkthrough — diffusion with the field, end to end

The sections above describe the components; this appendix tells the story of how they
work together, for a reader who wants intuition before (or instead of) equations.

### 10.1 Diffusion without the field

Take a real patch: 4,000 particles in a polycrystal. Imagine shaking it — every
particle displaced by a small random amount, then a larger one, over 1000 "timesteps"
until the configuration is indistinguishable from a Gaussian blob of points. That is
the forward process, and it is purely mechanical; nothing is learned. The noisy
configuration at any timestep can be written in closed form:

    x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps

with abar_t sliding from ~1 (barely noised) to ~0 (pure noise). The log-sigma schedule
(§3.1) controls how fast abar falls, tuned so the shaking passes gradually through
every physically meaningful length scale: jitter-sized displacements first, then
lattice-spacing-sized, then grain-sized, then everything.

The network learns exactly one skill: given the shaken configuration x_t and the noise
level t, predict the noise eps that was added. Training is: pick a random patch, pick a
random t, shake, predict, penalise MSE. Predicting eps is equivalent to predicting
where the clean particles were — the two are related by the formula above.

Generation runs the movie backwards. Start from a pure Gaussian blob (which looks
exactly like a fully-shaken sample, so the model is on familiar ground). Each step:
ask the model what noise would have produced the current configuration; use the answer
to compute the implied clean configuration x0_hat; take a small step from x_t toward
x0_hat (the DDPM posterior mean); re-add slightly less noise than before. Early on the
model's x0_hat guesses are vague ("particles roughly uniform in a disk") and the
re-added noise keeps options open. As the noise budget shrinks, the guesses commit to
progressively finer structure — overall density, then grains, then lattice rows, then
jitter-level positions. The sample crystallises out of noise, in the literal sense.

### 10.2 Why the field exists

Left alone, the model decides the grain layout by itself using only local neighbour
information — and it is structurally bad at that (grains come out too small, §2). The
field is how the mesoscale layout is handed to it.

The field is a 64x64 grayscale image F over the patch area, F(x, y) ~ local |psi6|:
bright = "this region is a well-ordered grain", dark line = "a grain boundary runs
here". It is deliberately coarse — smoothed over ~2 lattice spacings — so it dictates
where boundaries go but nothing about individual particles. A sketch the model must
paint over.

### 10.3 How the model reads the field (the key subtlety)

The field lives in space; the model's inputs are attached to particles; and particles
move, both during shaking and during generation. So the field enters as a lookup that
is redone at every step: each particle, at its current noisy position, bilinearly
interpolates the map there and receives that single number as an extra input feature
(alongside size, type, and the time embedding). A particle wandering through a dark
region is being told "you are in boundary territory" wherever it currently happens to
be; as particles settle over the reverse process, they keep re-reading the map at
their updated positions. The conditioning is glued to space, not to particles.

During training the map is computed from the clean patch, so the model learns: "given
where grains and boundaries are supposed to be, this is how particles arrange around
that." This is a far easier task than unconditional denoising — the mesoscale
decisions are already made; only local physics remains. And there is no train/test
cheat: the map is too coarse to encode individual positions, and at generation time we
supply a map of our own choosing, so the model sees the same kind of input in both
regimes (contrast with per-particle descriptors, §4).

In 15% of training steps the map is replaced by the null token (field value 0 with the
has_field flag set to 0), so the same network also learns to denoise blind. That yields
an unconditional mode and enables guidance.

### 10.4 One generation step with everything on

Given a target map (from the stage-1 generator, a real patch, or drawn by hand), each
reverse step is:

1. Build the kNN graph on the current noisy positions.
2. Look up the map value at every particle's current position.
3. Run the model twice: with the real map values (eps_cond) and with the null token
   (eps_null).
4. Combine with classifier-free guidance:
       eps_hat = eps_null + w * (eps_cond - eps_null),   w = 2 currently.
   The bracketed difference is the pure "influence of the map": everything the model
   would predict anyway (lattice physics, spacing, density) appears identically in
   both evaluations and cancels. w > 1 pushes the sample harder in the map-following
   direction than the model would naturally go. If the sample ignores the map, raise
   w (typical useful range 1.5-4; too high causes over-sharpened, artificial order).
5. Compute x0_hat from eps_hat; radially threshold any particle implied to be absurdly
   far out; in the last quarter of the process, apply the soft-disk repulsion force to
   x0_hat so no two particles interpenetrate and the large defects push neighbours out
   into proper cavities.
6. Step toward x0_hat, re-add the scheduled (smaller) noise, re-centre the CoM,
   continue.

### 10.5 The interplay over time

At high noise the map's influence is broad: it biases where disorder will eventually
live, steering coarse density and the future grain skeleton. In the mid-process — when
the noise scale matches the grain scale — the map does its main work: particles in
bright regions are pulled into coherent lattice patches; particles under dark lines
are held in frustrated arrangements between mismatched neighbours. At low noise the
map barely matters any more; local physics and the repulsion force polish spacing and
jitter. The end product is new particles, in a new microscopic arrangement, realising
the layout the map prescribed.

One-sentence summary of the architecture: **the map decides where order and disorder
go; the diffusion model decides what order and disorder look like, one particle at a
time; and guidance is the volume knob between the two.**

## 11. Current hyperparameters

| item | value |
|---|---|
| patch | square, ~4900 particles (70×70 diameters), 5 random crops/box |
| k / layers / hidden | 12 / 6 / 128 (+ per-layer FiLM, cond_dim 256) |
| diffusion steps / schedule | 1000 / logsigma, σ_eff ∈ [0.001, 60] |
| coord_scale (square patches) | ~23 (stored in checkpoint) |
| RBF | 24 Gaussians, [0, 0.4] normalised |
| optimizer | AdamW, lr 2e-4 (1e-4 resume), wd 1e-4, clip 1.0 |
| EMA decay | 0.999 |
| field | 64×64, smooth ≈2 spacings, dropout 15%, CFG w=2 |
| repulsion | strength 0.5, last 25% of steps |
| corrector | 1 Langevin step, SNR 0.15 |
| x̂₀ threshold | radial, 4.0 normalised |
| map generator | UNet ~1M params, 400 cosine steps, 64×64 |
