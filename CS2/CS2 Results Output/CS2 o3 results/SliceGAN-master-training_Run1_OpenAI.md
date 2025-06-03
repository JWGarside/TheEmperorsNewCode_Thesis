# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-training
**Analysis Date:** 2025-05-25

## Analysis Results

1. Brief paper summary and core claims  
• SliceGAN is a GAN-based “dimensionality-expansion’’ method that learns from one representative 2-D micrograph (or three orthogonal images for anisotropic media) and synthesises statistically similar 3-D volumes.  
• Key ideas/claims  
  – Generator is fully 3-D; a slicing operation converts the generated cube into many 2-D images that are passed to conventional 2-D discriminators.  
  – Uniform-information-density constraint on every transpose-convolution layer (k = 4, s = 2, p = 2) avoids edge artefacts and allows arbitrarily large volumes.  
  – Uses a WGAN-GP loss; typical training volumes are 64³ voxels from a 4×4×4 latent seed.  
  – For isotropic materials the same discriminator is trained on the full set of 3 l slices in x,y,z; anisotropic training uses three separate discriminators, one per orientation.  
  – After ≈4 h training, generators can create 10⁸-voxel volumes in a few seconds whose structural statistics match those of the real data.

2. Implementation assessment (files shown above)  
• run_slicegan.py: builds networks, launches training (model.train) or inference.  
• networks.py:   – Generator: 5 transpose-conv layers (k=4, s=2, p=[2,2,2,2,3]) → 64³ output, final soft-max.   – Uses a 4×4×4 latent tensor, exactly as described in the paper.   – Implements the required {4,2,2} transpose-conv rule, so the uniform-density design is respected.  
• model.py (training loop):   – Creates three discriminators (or one reused 3× for isotropic).   – Noise shape (B, nz, 4,4,4).   – WGAN-GP loss exactly as in paper.  
• preprocessing.py: extracts many random 2-D patches (32 × 900 = 28 800) from the training image(s) to form the batches used by the discriminators. One-hot encoding for n-phase data, matching the manuscript.  
Overall, most methodological details in the manuscript are implemented faithfully: latent-vector size, network depths, transpose-conv parameters, WGAN-GP, isotropic/anisotropic option, one-hot encoding, training hyper-parameters, and inference procedure.

3. Observed discrepancies  

| # | Description | Impact | Classification |
|---|-------------|--------|----------------|
|1 | Paper states that during each discriminator update every generated cube is sliced into the full set of 3 l planes; code (model.py, lines 50-64) takes **only a single central slice** `fake_data[:, :, l//2, :, :]` for all three orientations (permutation not applied). Generator updates do use many slices, but the discriminator sees far fewer views than reported. | May reduce the amount of information reaching D and change training dynamics; core idea (multi-slice supervision) is not fully reproduced. | Critical |
|2 | Paper specifies m_G = 2 m_D (larger generator batch to balance many fake slices). Code uses `batch_size == D_batch_size == 8`; the balancing mechanism is therefore absent. | Likely influences convergence speed but not the fundamental architecture. | Minor |
|3 | Paper’s explicit slicing permutation for anisotropic case is absent; code sets a Boolean `isotropic` flag and, if true, collapses the three discriminators into one. That matches the isotropic case, but anisotropic training in Algorithm S1 would require three **different** real data loaders; code would need three separate file paths, but the permutation variables (d1,d2,d3) are not used on the discriminator side. | Anisotropic training, as described in the paper, cannot be reproduced without code modification. | Critical for anisotropic results |
|4 | Generator in networks.py adds an up-sampling + 3-D conv (“rcconv”) stage not discussed in the paper. | Architectural difference is small; unlikely to change headline claims. | Minor |
|5 | Default training epochs, learning rates, etc. are not listed in the manuscript but are hard-coded in model.py; this is a documentation rather than methodological gap. | None on reproducibility once code is available. | Cosmetic |

4. Overall reproducibility conclusion  
The provided code embodies almost all of SliceGAN’s published architectural principles (3-D generator, WGAN-GP, 4³ latent seed, information-density rule) and therefore should allow readers to reproduce the central qualitative claim: generation of realistic 3-D microstructures from 2-D data.  
However, two critical discrepancies remain:

• The discriminator is trained on only one slice per cube instead of the full 3 l slice set, deviating from the paper’s stated algorithm.  
• The anisotropic-material training procedure described in the Supplementary Algorithm (separate discriminators with orientation-specific real images) is not fully realised in the released code.

If a user’s goal is to replicate the isotropic results presented in the paper, the code will likely succeed (and qualitative examples in the repository corroborate this). Reproducing the precise training regime and the anisotropic demonstrations would require moderate code changes.