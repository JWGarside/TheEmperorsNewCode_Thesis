# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-inputvector
**Analysis Date:** 2025-05-25

## Analysis Results

1. Brief paper summary and core claims  
• “Generating 3-D structures from a 2-D slice with GAN-based dimensionality expansion” introduces SliceGAN, a Wasserstein-GAN that is trained only on 2-D micrographs yet can synthesise statistically realistic 3-D volumes.  
• Key ideas  
 – A 3-D generator (G) is paired with a 2-D discriminator (D). Every generated cube is sliced along x, y, z; the resulting 2-D images are fed to D while the 2-D training micrograph supplies the “real” batch.  
 – For anisotropic media three perpendicular micrographs and three discriminators are used.  
 – Uniform-information-density: transpose–convolution parameters are constrained to {k=4, s=2, p=2} (or compatible sets) so that all voxels are produced by the same number of kernel parameters, eliminating edge artefacts and permitting generation of arbitrarily large volumes.  
 – The latent tensor fed to G has a spatial size of 4×4×4 so that kernel overlap is learned during training; after convergence the latent size can be varied to output larger volumes “for free”.  
 – With these choices SliceGAN is reported to generate 108-voxel volumes in seconds that match real 3-D datasets on two-point correlations, tortuosity, TPB density, etc.  

2. Implementation assessment  
Execution flow (run_slicegan.py → networks.py → model.py):  
• run_slicegan.py sets hyper-parameters, builds nets with networks.slicegan_rc_nets, and either trains (model.train) or loads a trained G for inference.  
• preprocessing.batch() converts a 3-D tiff (or 2-D images) to one-hot 2-D tiles (size 64×64 by default).  
• model.train():  
 – Builds 3 discriminators and one generator.  
 – For every training step: samples noise, produces a 3-D fake volume, permutes it to obtain the l (=64) axial slices, and feeds them to the relevant D together with real 2-D tiles.  
 – Uses WGAN-GP loss (util.calc_gradient_penalty).  
 – Critic/G loops, logging, checkpointing and visualisation follow the paper recipe.  
• Generator architecture (networks.slicegan_rc_nets)  
 – 5 transpose-conv3-D layers with the user-defined lists gk,gs,gp (default 4,2,2) followed by a trilinear up-sample and a 3×3×3 Conv3D, final soft-max.  
• Discriminator is a 5-layer 2-D Conv net that mirrors df,dk,ds,dp.  

Overall, the major algorithmic elements (3-D→2-D slicing, WGAN-GP, one-hot encoding, kernel settings to avoid edge artefacts) appear in the code exactly as described in the paper.

3. Categorised discrepancies  

| # | Paper description | Code implementation | Category | Comment |
|---|-------------------|---------------------|----------|---------|
| 1 | Latent tensor has spatial size 4×4×4 to learn overlap and enable arbitrary output size. | In model.train `lz = 1` and all noise tensors are created with shape *(B, nz, 1, 1, 1)*. | Critical | A 1×1×1 latent breaks the “kernel-overlap” motivation in the paper and contradicts the stated requirement `s<k`.  It also makes the first transpose-conv output size 0 with the default (k=4, s=2, p=2) formula, so the network would fail without further code changes. |
| 2 | Generator uses only transpose-conv layers with parameters chosen from {4,2,2}. | `slicegan_rc_nets` inserts a trilinear `Upsample` plus a 3-D convolution after the last transpose-conv, and its `size = (int(x.shape[2]-1,)*2, …)` line contains a trailing comma that makes `int()` receive a tuple, raising a `TypeError`. | Critical | This syntactic error prevents the generator from running in its present form; in addition, the extra up-sample stage is not mentioned in the paper. |
| 3 | m_G = 2 m_D as a heuristic for balancing G/D updates. | Code uses the same minibatch size (8) for G and D and critic_iters = 5. | Minor | Affects training stability/ speed but not the core methodology. |
| 4 | Edge-uniformity requires padding ≥ k-s (=2). | run_slicegan sets `gp = [2,2,2,2,3]`; final layer uses p=3, k=4, s=2 thus still satisfies the rule, albeit not documented in the paper. | Cosmetic | Still conforms to the constraint. |
| 5 | Paper emphasises ability to specify periodic boundaries by duplicating the first two latent slices. | Code provides a `periodic` option in util.test_img (inference only) but the training script never enforces periodicity during training. | Minor | Periodicity in the final volume may deviate from the claim unless the user enables it manually. |

4. Overall reproducibility conclusion  
The repository contains most of the components needed to reproduce SliceGAN, and the high-level training loop follows the algorithm described in the paper.  However, two critical issues—(i) the latent tensor size hard-coded to 1×1×1 instead of 4×4×4 and (ii) a syntax error in the generator’s up-sampling block—prevent the code, as currently provided, from running and therefore from reproducing the paper’s quantitative results without modification.  After these fixes (and with hyper-parameters adjusted to those reported in the manuscript) the implementation should reproduce the main claims, but out-of-the-box reproducibility is not yet achieved.