# TODO

1. ~~Find `g(r)` minima to set the cutoff radius~~ (done; graphs now use kNN, k=12)
2. ~~Write the `build_node_scalars` function~~ (done; now used for *evaluation only*, not conditioning)
3. Run `overfit_test.py` on a real patch — confirm the sampled patch is lattice-like
4. Full training run: `python train_patched.py` (uses only size/type conditioning + coordinate normalisation)
5. Sample with `python sampling.py` (reads architecture + coord_scale from the checkpoint)
6. Write an evaluation script: compare g(r), psi6/psi4 distributions, and Voronoi
   statistics (via `node_features.build_node_scalars`) between generated and real patches
7. Possible later improvements:
   - batch several patches per gradient step (block-diagonal graphs) for faster epochs
   - condition on a target grain-boundary density / psi6 map to steer generation
   - periodic boundary handling to generate full boxes by stitching patches
