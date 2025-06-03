# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-anisotropic
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• “Generating 3-D Structures from a 2-D Slice with GAN-Based Dimensionality Expansion” introduces SliceGAN, a Wasserstein-GAN-GP that  
  – uses a 3-D generator and a 2-D discriminator; the generated volume is sliced into all axial planes so that only 2-D images are ever shown to the discriminator.  
  – guarantees uniform information density in the generator by restricting every ConvTranspose3d layer to k = 4, s = 2, p = 2 (or equivalent) and by feeding a 4 × 4 × 4 latent tensor.  
  – needs only one 2-D training micrograph for isotropic media or three orthogonal micrographs plus three discriminators for anisotropic media.  
  – can synthesise arbitrarily large, statistically faithful 3-D microstructures in seconds.  
The paper supplies a reference architecture (Table 1), explicit training hyper-parameters (mD, mG, nD, λ etc.) and evaluation on seven datasets.

2. Implementation assessment  
• Package structure: `run_slicegan.py` → network construction (`slicegan/networks.py`) → training loop (`slicegan/model.py`).  
• Core ideas are present:  
  – Generator is 3-D; discriminator is 2-D.  
  – Training loop slices the fake volume by `permute(...).reshape(l*B, C, l, l)` and applies a WGAN-GP loss.  
  – Uniform-kernel rule (k=4, s=2) is used by default in the example script.  
  – Isotropic vs. anisotropic cases are distinguished by the number of training images supplied.  
• Default script (`run_slicegan.py`) chooses `slicegan_rc_nets`, whose generator has 4 ConvTranspose3d stages followed by an **upsample + 3×3 Conv (rcconv)**; this differs from the paper’s pure-transpose architecture but preserves the dimensionality-expansion idea.  
• Training hyper-parameters in `model.py`: batch_size = 8, critic_iters = 5, λ = 10, learning rates = 1e-4; these are reasonable although not identical to the paper’s mG = 2 mD guideline (here both are 8).  
• Utility functions reproduce figures, save generators, and export .tif volumes exactly as described.

3. Discrepancies  

| # | Description | Impact | Class |
|---|-------------|--------|-------|
| 1 | In the discriminator loop the code forcibly sets `netD = netDs[0]` for **all three directions**, even when `isotropic == False`. As a result only one discriminator is actually trained for anisotropic data, contrary to Algorithm S1. | The anisotropic experiments in the paper may not be reproducible without fixing this line. | Critical |
| 2 | Default generator used in the script adds an `Upsample + Conv3d` (“rcconv”) stage not mentioned in the paper and uses filter sizes [1024, 512, 128, 32] instead of [512, 256, 128, 64]. | Could change memory use and visual quality but keeps the core pipeline; isotropic results are still likely. | Minor |
| 3 | `laysd = 6` but `dp` length is 5, so the last Conv2d layer in the discriminator is silently dropped (zip stops at shortest). Paper specifies 6 layers. | Slight capacity reduction; unlikely to invalidate claims. | Minor |
| 4 | Batch-size ratio mG = 2 mD stated in paper is not implemented (both are 8). | May affect convergence speed not principle. | Minor |
| 5 | README refers to `train.py`; actual file is `model.py`. | None on results. | Cosmetic |

4. Overall reproducibility conclusion  
The released code captures the essential SliceGAN mechanism (3-D → 2-D slicing, WGAN-GP training, information-density constraints) and should let users reproduce the isotropic experiments after adjusting paths and running `python run_slicegan 1`.  
However, a hard-coded override in the training loop means anisotropic results reported in the paper cannot be faithfully reproduced without modifying two lines so that distinct discriminators are used for the x, y and z slices. Apart from this bug and several minor architectural/hyper-parameter deviations, the implementation is largely consistent with the paper.