# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-anisotropic
**Analysis Date:** 2025-05-25

## Analysis Results

1. Brief paper summary and core claims  
• The paper introduces SliceGAN, a generative adversarial‐network framework that synthesises 3-D micro-structures from only one (isotropic) or three (anisotropic) 2-D slices.  
• Key methodological ingredients  
 – A 3-D generator and a 2-D discriminator linked by an on-the-fly “slicing” operation so that every generated voxel plane is judged against real 2-D images.  
 – Uniform-information-density transpose-convolution design rules (kernel size k, stride s, padding p) to avoid edge artefacts and allow arbitrarily large output volumes.  
 – Wasserstein-GAN with gradient penalty (WGAN-GP) for stable training; generator input z of spatial size 4×4×4 to guarantee kernel overlap and periodic tiling every 32 planes.  
• Claims: SliceGAN can (i) reproduce visual appearance of many materials, (ii) match statistical metrics (volume fraction, TPB, two-point correlations, effective diffusivity) and (iii) generate 10^8-voxel samples in seconds once trained.

2. Implementation assessment  
Execution flow (run_slicegan.py → slicegan/model.py → slicegan/networks.py):

a. Data pipeline  
 • preprocessing.batch() draws 32 × 900 random l×l slices per orientation; one-hot encoding for n-phase data as in paper.  
 • For 3-D TIFF training images it chooses a random plane from each axis, matching the paper’s isotropic assumption.

b. Networks  
 • networks.slicegan_rc_nets() builds a 3-D ConvTranspose generator (5 layers by default).  
 • Default kernel-stride-padding lists are gk=dk=[4,…], gs=ds=[2,…], gp=[2,2,2,2,3], dp=[1,1,1,1,0] – i.e. {k,s,p}={4,2,2} except for the last layer where p=3 to recover 64³ volume. This obeys the paper’s “uniform-information-density” rules.  
 • Generator finishes with soft-max (n-phase) or tanh (colour/greyscale) exactly as stated.  
 • The first input tensor z has dimension (nz,4,4,4) (lz=4), again matching the paper.  
 • An extra up-sample + 3-D conv (“rcconv”) is implemented to halve the periodicity (32-voxel period), an optimisation mentioned in the supplementary material.

c. Training loop (model.train)  
 • Implements WGAN-GP with λ=10, critic_iters=5, identical batch sizes of eight.  
 • Generator loss is –D(fake); discriminator loss is D(fake)–D(real)+GP, identical to Algorithm 1.  
 • For isotropic data len(real_data)==1 so only one discriminator is effectively used; for anisotropic three discriminators are created, following Algorithm S1.  
 • fake volumes are reshaped to l×batch 2-D slices along each axis, so each slice is seen by D exactly as described.

d. Inference utilities (util.test_img)  
 • Loads trained G, draws (nz,lf,lf,lf) noise, optionally enforces periodic borders, saves TIFF – aligned with the paper’s discussion of large/periodic output.

Overall, core algorithms and parameter choices match the methodological description.

3. Discrepancies

| # | Observation | Type | Comment |
|---|-------------|------|---------|
|1|Paper Table 1 first ConvTranspose layer uses {k,s,p} = {4,2,2} and 512 output channels; default code uses 1024 output channels|Minor|Affects memory but not the principle; filter numbers are optimisation choices.|
|2|Paper states discriminator is shown “all 64 slices in each direction”; code draws l×batch slices (l=64) **per batch** but over mini-batches, so every slice is still eventually seen but not necessarily within a single optimisation step|Minor|Slower convergence at worst; does not invalidate method.|
|3|Uniform-density rule requires p ≥ k–s (=2). Code uses p=3 in last generator layer; still satisfies rule and keeps output size|Cosmetic||
|4|Supplement points out 32-voxel periodicity; code adds rcconv+Upsample which is not explicitly discussed in main text|Cosmetic|Enhancement, not contradiction.|
|5|Default training epochs (100) and batch sizes (8) differ from example numbers in manuscript (the paper quotes ~4 h on Titan Xp which would correspond to larger batches); runtime only, not scientific.|Cosmetic||

No critical discrepancies that would block reproduction were found.

4. Overall reproducibility conclusion  
The released implementation follows the paper very closely: data handling, architecture (transpose-conv generator with uniform-information-density settings), training loss (WGAN-GP), and inference procedure all correspond to the documented methodology. Observed differences are limited to non-essential hyper-parameters (filter counts, padding in last layer, number of slices per optimisation step) or added convenience features.  

Provided that suitable training images are supplied, a reader should be able to replicate the core qualitative and quantitative results claimed in the paper with this code base.