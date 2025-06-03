# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-training
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• The paper introduces SliceGAN, a GAN architecture that can “expand the dimensionality” of micro-structural data, i.e. learn from a single representative 2-D slice (or three mutually-orthogonal slices for anisotropic cases) and synthesise statistically consistent 3-D volumes.  
• Key ingredients reported in the manuscript:  
 – A 3-D generator built only from transpose–convolution layers using the parameter set {k=4, s=2, p=2} (last layer {4, 2, 3}) to guarantee uniform information density.  
 – Input latent tensor of size 64 × 4 × 4 × 4 to enforce kernel overlap and permit arbitrary output size.  
 – Training procedure (Algorithm 1): for every generated volume the generator is sliced into all l planes in x, y and z (3 l images). Each slice is fed to a 2-D discriminator. mG = 2 mD is recommended to balance the much larger number of discriminator updates. Wasserstein-GP loss (λ=10, nD=5).  
 – For isotropic data one discriminator is reused for the three orientations; for anisotropic data three separate discriminators are trained.  
 – After training, volumes as large as 108 voxels can be produced in a few seconds and have nearly identical tortuosity, TPB density and two-point-correlation functions to the ground-truth data.

2. Implementation assessment  
Execution flow (run_slicegan.py → networks.py → model.py)  

• Network architecture  
 – Default run (`run_slicegan`) constructs the generator/discriminator pair through `networks.slicegan_rc_nets`.  
 – gk/gs/gp are [4,4,4,4,4] / [2,2,2,2,2] / [2,2,2,2,3] respectively, matching the paper’s kernel/stride/padding rules.  
 – Filter sizes are [32,1024,512,128,32,3] which also mirrors the paper (Table 1).  
 – However the “_rc_” variant replaces the final transpose-conv with an Upsample (trilinear) followed by a 3 × 3 × 3 Conv (“resize-convolution” route discussed—but not adopted—in the paper’s SI).  
• Training loop (`model.train`)  
 – Three discriminator instances are always created. When `isotropic=True` only the first one is used.  
 – Mini-batch sizes: batch_size = D_batch_size = 8 (mG = mD, not 2 mD as recommended).  
 – WGAN-GP with λ = 10, critic_iters = 5 exactly as in paper.  
 – Dataset construction (`preprocessing.batch`) randomly samples 32 × 900 patches per orientation (same order of magnitude as stated).  
• Slicing strategy in code  
 – For discriminator steps only a **single central slice** (`fake_data[:, :, l//2, :, :]`) of the generated volume is used, not the full set of 64 slices per axis that Algorithm 1 specifies.  
 – During generator updates a permutation/reshape feeds **all** slices to the discriminator (`permute(...).reshape(l*batch,…)`), so the two nets see different slice distributions.  
• Hyper-parameters (learning rate, β1/β2 etc.) match those printed in the paper.

3. Discrepancies

| # | Description | Severity |
|---|-------------|----------|
| 1 | `slicegan_rc_nets` generator contains a syntax error: `size = (int(x.shape[2]-1,)*2, …)` – the trailing comma makes `int()` receive a tuple and will raise `TypeError`, preventing training when the “rc” network is used. | Critical |
| 2 | Default script uses the “resize-convolution” generator (`slicegan_rc_nets`) whereas the paper states that the production models use only transpose-convolutions because resize-conv is memory-heavy. | Critical (different architecture could affect claimed results) |
| 3 | Discriminator is trained on one central slice per volume, not on every slice as required by Algorithm 1. | Critical (core training procedure not faithfully reproduced) |
| 4 | mG = mD in code; the paper recommends mG = 2 mD to balance slice counts. | Minor |
| 5 | Batch-norm/instance-norm choice differs from text (paper lists BatchNorm3d; “rc” variant mixes BatchNorm and a final Conv3d). | Minor |
| 6 | Uniform-information-density rule is slightly violated in last generator layer (k = 4, s = 2, p = 3 → p>k-s so acceptable, but paper last layer is still transpose-conv, not resize-conv). | Minor |
| 7 | Code hard-codes training to 100 epochs; paper does not specify exact number for each example but implied early stopping on convergence. | Cosmetic |

4. Overall reproducibility conclusion  
The repository contains all major components (data preprocessing, 3-D generator, 2-D discriminator, WGAN-GP training loop) needed to reproduce SliceGAN; however, the default pipeline deviates from the manuscript in three critical ways:

1. It selects a different generator architecture (resize-convolution) that the paper explicitly argues against.  
2. A syntax error in that generator will crash execution.  
3. The discriminator is not exposed to the full set of slices, undermining the central “slice-based” learning strategy.

Unless a user manually switches to `networks.slicegan_nets`, fixes the size-computation bug, and modifies the training loop to feed every slice to the discriminator, the provided code cannot reproduce the results reported in the paper. Therefore, the current code base is **not fully reproducible** for the paper’s core claims without non-trivial corrections.