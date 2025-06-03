# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-inputvector
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• SliceGAN is a GAN-based “dimensionality-expansion” framework that learns from one (isotropic case) or three (anisotropic case) 2-D micrographs and synthesises statistically-consistent 3-D volumes.  
• Key methodological points in the paper  
  – Architecture: 3-D generator, 2-D discriminator(s). A generated 3-D cube of edge-length l is sliced into the l planes in x, y, z and all 3 l images are given to the discriminator(s).  
  – Loss: WGAN-GP with λ = 10, critic/generator update ratio nD = 5, Adam (lr = 1 × 10-4, β = (0,0.9)).  
  – Uniform-information-density rule for transpose convolutions: kernel 4, stride 2, padding 2 for every layer except the output layer (padding 3).  
  – Latent seed z has spatial size 4×4×4 during training so that the first transpose convolution already contains kernel overlap; this is claimed to (i) avoid edge artefacts and (ii) allow arbitrary output size at inference by enlarging the seed.  
  – Batch sizes: generator batch twice the discriminator batch (mG = 2 mD). Example values in text are mD = 8, mG = 16.  
  – Output cube generated during training is 64³ voxels; volumes up to 108 voxels are reported at inference.  

2. Implementation assessment (main flow)  
run_slicegan.py ⟶ slicegan.networks.slicegan_rc_nets()  
 • generates Generator/Discriminator classes from parameter lists supplied in run_slicegan.py  
 • Generator: 5 ConvTranspose3d layers (kernels 4, stride 2, padding given by gp list), BatchNorm after each, final 3-D conv (rcconv) + soft-max.  
 • Discriminator: 2-D Conv layers matching description in paper.  
 • Training loop (slicegan.model.train):  
  – uses three independent discriminators (or one reused three times if isotropic==True).  
  – slicing is implemented with permute + reshape exactly as described.  
  – WGAN-GP loss with λ = 10, critic_iters = 5.  
  – Adam optimiser with lr = 1e-4, β = (0.9,0.99).  
  – Batch sizes fixed at 8 (both G and D).  
  – Latent noise tensor shape is (batch, nz, 1,1,1); lz is hard-coded to 1.  
 • Pre-processing builds 32 × 900 random 2-D slices per axis (matches “64 slices” only partly).  

3. Discrepancies found  

| # | Description | Paper | Code | Impact | Class |
|---|-------------|-------|------|--------|-------|
| 1 | Latent seed spatial size during training | 4×4×4 | 1×1×1 (`lz = 1`) | First generator layer has no kernel overlap → violates the uniform-density argument; enlarging seed at inference may cause artefacts the paper claims to avoid. | Critical |
| 2 | First transpose-conv padding | Should be `p=0` for 1-voxel input or `p=2` for 4-voxel input (paper always gives consistent set) | gp list hard-coded `[2,2,2,2,3]` while input depth is 1 | With input=1 and padding=2 PyTorch formula gives output size 0 ⇒ layer would fail; training would crash unless user manually edits parameters. | Critical |
| 3 | Optimiser β parameters | β = (0,0.9) recommended in WGAN-GP literature and stated in paper | β = (0.9,0.99) | Slower / less stable convergence; does not invalidate method but can hinder reproduction of reported quality. | Minor |
| 4 | Generator variant (`slicegan_rc_nets`) adds up-sample + extra conv not mentioned in paper; size computation line contains syntax error (`int(x.shape[2]-1,)`) | Not in manuscript | Present in code; line as written raises `TypeError` | Code cannot run without manual fix; architecture different from paper. | Critical |
| 5 | mG vs mD ratio | mG = 2 mD | Both batch sizes set to 8 → ratio = 1 | Affects training dynamics but not conceptual algorithm. | Minor |
| 6 | Number of discriminator slices per iteration | Paper argues for showing at least 64 slices per axis. | Only l × batch slices (64×8=512) are formed, but they come from **one** 3-D cube; discriminator batch of *real* images is only 8 per orientation, not 64. | Could reduce effectiveness but does not prevent training. | Minor/ Cosmetic |
| 7 | Wording vs code variable names (e.g. “Lambda” not “λ”) | N/A | Cosmetic | No effect | Cosmetic |

4. Overall reproducibility conclusion  
The released code base contains several **critical inconsistencies** with the methodology described in the paper. In its current form:

• The generator cannot initialise correctly with the provided kernel/stride/padding when the latent cube is of size 1; execution will raise an error unless the user manually edits either `lz` or `gp[0]`.  
• The latent-size choice contradicts the paper’s central argument on avoiding edge artefacts, so even after fixing the dimension mismatch the trained model would differ from the one evaluated in the manuscript.  
• An additional generator variant (`rcconv` & up-sample) that is not documented in the paper contains a syntactic error that blocks execution.  

Minor deviations (optimizer β, batch-ratio) further distance the implementation from the reported setup, but they are secondary compared with the structural issues above.

Therefore, **the code is currently not directly runnable and does not faithfully reproduce the experimental setup claimed in the paper without non-trivial manual corrections**.