# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-checkerboard
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary & core claims  
• SliceGAN is a GAN-based “dimensionality-expansion’’ framework that learns from one (isotropic) or three (anisotropic) 2-D micrographs and synthesises statistically-consistent 3-D volumes.  
• Key methodological points in the paper  
  – 3-D generator + 2-D discriminator.  Generated volumes are cut into 2-D slices (64 per axis normally) before being passed to the discriminator.  
  – Wasserstein-GP objective with nD = 5 discriminator steps, λ = 10, Adam(β1 = 0.9,β2 = 0.99).  
  – Generator architecture based on strided 3-D transpose-convolutions with uniform-information-density rules: kernel k = 4, stride s = 2, padding p = 2 in every layer (k mod s = 0, s<k, p≥k-s).  Latent tensor is size (Cz=32,4,4,4) so that the network already “knows” about overlap; this allows arbitrary output size at inference.  
  – Batch-size relation mG = 2 mD (generator trained on twice as many 2-D slices as the discriminator sees from real data).  
  – For anisotropic cases, three independent discriminators are used, each trained on slices perpendicular to its own axis.  
  – Colour / greyscale outputs use tanh; n-phase outputs use soft-max one-hot coding.

2. Implementation assessment  
Execution path (run_slicegan.py → networks.slicegan_rc_nets → model.train):  
• run_slicegan.py builds a generator/discriminator pair with slicegan_rc_nets, then calls model.train.  
• slicegan_rc_nets creates       – Generator: 5 ConvTranspose3d layers followed by an up-sample + 3×3×3 Conv3d “recoding” layer.  
      – Discriminator: 5 Conv2d layers.  
• Architectural parameters are taken from run_slicegan.py:  
      gk =[4,4,4,4,4], gs =[3,3,3,3,3], gp =[1,1,1,1,1]  
      dk =[4,4,4,4,4,4], ds =[2,2,2,2,2,2], dp =[1,1,1,1,0]  
• model.train implements the slice-based WGAN-GP exactly as described (three orientations, isotropic shortcut when only one training micrograph is supplied; critic_iters = 5, λ = 10, Adam β’s match the paper).  
• Gradient-penalty code and slice permutation/reshaping are faithful to the manuscript.  
• Dataset preprocessing supports colour, greyscale, 2-phase and n-phase as described, with 32 × 900 random crops per orientation (≈ 30 k images) the same as used in the paper.

3. Detected discrepancies  

| # | Item | Paper | Code | Impact | Classification |
|---|------|-------|------|--------|----------------|
|1| Generator stride / padding | k = 4, s = 2, p ≥ 2 to guarantee uniform information density | gs = 3, gp = 1 in run_slicegan.py | Violates k mod s = 0 and p ≥ k-s rules; can re-introduce checker-board / edge artefacts the paper claims to eliminate | **Critical** for reproducing the high-quality edge statistics shown in the paper |
|2| mG = 2 × mD | Paper explicitly states generator slice-batch twice discriminator | Code uses batch_size = D_batch_size = 8, hence mG = mD | May alter training dynamics and image fidelity but does not change the core algorithm | Minor |
|3| Separate discriminators for anisotropy | Paper: 3 discriminators for anisotropic data | Code implements 3 discriminators but run_slicegan supplies only one training image; isotropic shortcut collapses to a single discriminator. | Behaviour matches isotropic case tested in run script; no discrepancy. | Cosmetic |
|4| Latent-tensor spatial size | Paper uses (4×4×4) to encode kernel overlap from the first layer | run_slicegan keeps that (lz=4) – matches paper | – | – |
|5| Padding in discriminator last layer | Paper’s discriminator example ends with 4,2,0; code uses dp=[1,1,1,1,0] (padding 1 in early layers) | Small architectural difference; does not affect concept | Cosmetic |
|6| Uniform-density rule re-emphasised in text | Not enforced programmatically; user can accidentally violate (see #1) | Lack of automatic check | Minor |

4. Overall reproducibility conclusion  
The released codebase captures the essential SliceGAN training algorithm (slice-based WGAN-GP, anisotropy extension, one-hot outputs, latent 4×4×4 input).  Apart from hyper-parameter defaults, most implementation details match the manuscript.

However, the default network hyper-parameters supplied in the example run_slicegan.py (stride = 3, padding = 1) break the uniform-information-density conditions that the paper argues are crucial.  Using these defaults is therefore unlikely to reproduce the edge-quality and quantitative metrics reported in the paper.  If the user edits gs to 2 and gp to 2 (or uses the parameter set stated in the paper), the implementation should reproduce the core results.

Hence reproducibility is feasible but not “out-of-the-box”; one critical parameter adjustment is required.