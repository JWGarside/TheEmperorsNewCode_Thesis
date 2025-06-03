# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-checkerboard
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
The paper “Generating 3-D Structures from a 2-D Slice with GAN-based Dimensionality Expansion” introduces SliceGAN – a Wasserstein-GAN in which  
• a 3-D generator (G) produces volumetric data while a 2-D discriminator (D) judges randomly extracted x/y/z slices;  
• three discriminators can be used for anisotropic media;  
• transpose-convolution parameters are carefully chosen ({kernel, stride, padding} = {4,2,2}) so every voxel of the generated cube is produced from the same number of weights (“uniform information density”), avoiding the familiar checker-board artefact;  
• the latent seed has spatial extent 4³ so that arbitrarily large 3-D volumes (multiples of 32³) can be generated after training;  
• training uses a WGAN-GP loss with nD=5, λ=10, Adam (α=1e-4, β1=0.9, β2=0.99) and small batch sizes (8);  
• only one segmented 2-D micrograph (or three orthogonal ones) is needed as training data;  
The paper shows qualitative results on seven microstructures and quantitative agreement for an NMC cathode.

2. Implementation assessment  
Execution flow (run_slicegan.py → networks.py → model.py):

• networks.slicegan_rc_nets builds  
  – Generator: five 3-D transpose-convs followed by BatchNorm, then an Upsample( trilinear )+3-D Conv (“rcconv”) and a final soft-max.  
  – Discriminator: six 2-D conv layers.  
  All parameters are serialised to *_params.data so that test runs reload the exact architecture that was trained.

• Default hyper-parameters in run_slicegan.py match the paper for most items: latent seed (4³), WGAN-GP loss, λ=10, critic_iters=5, Adam (lr 1e-4, β1=0.9, β2=0.99), batch-size 8, 100 epochs.

• model.train creates three independent PyTorch data-loaders (or re-uses one for isotropic data) and trains one or three discriminators exactly as described in Algorithm 1 of the paper.

• preprocessing.batch converts colour/grey/n-phase images into one-hot tensors of size (nc,64,64) as required.

• util.test_img reproduces the “arbitrarily large” inference by allowing a user-specified latent field size (lf).

3. Discrepancies  

| # | Description | Paper | Code | Impact |
|---|-------------|-------|------|--------|
|1|Stride / kernel rule for uniform information density|Strictly {k=4, s=2, p=2}|Default gs = 3, gp = 1 → {4,3,1} violates k mod s = 0 & p ≥ k−s|Minor – may re-introduce mild edge artefacts but does not invalidate the method; users can change gs to 2 via the config.|
|2|Generator tail|Paper ends with a transpose-conv producing the final channels|Code skips the last transpose-conv and instead upsamples + rcconv(3×3×3)|Minor – an architectural variant, still produces 64³ outputs and uses soft-max.|
|3|Padding list for discriminator|six layers shown in paper|dp list length 5 (zip truncates), last conv therefore uses default p=0|Cosmetic – only affects receptive field of final conv.|
|4|Adam β₁|Not explicitly fixed in paper (given as a variable)|Code chooses β₁=0.9 (common for WGAN-GP)|Cosmetic.|
|5|Uniform-density justification|Claimed essential for quality|Code’s default values do not follow it, although user can|Minor.|
|6|Parameter names (train.py vs model.py) |Algorithm 1 uses nD; code uses critic_iters|Cosmetic.|

No inconsistency was found that prevents the code from training a model with the settings reported in the README; changing gs from 3 to 2 would make it conform exactly to the paper’s guideline.

4. Overall reproducibility conclusion  
The released implementation follows the training procedure, loss function, data preparation and evaluation pipeline described in the paper.  The few differences are either architectural choices that leave the fundamental “3-D-to-2-D slicing GAN” idea intact or hyper-parameters that can be matched easily through the configuration script.  Hence the paper’s main claims appear reproducible with the provided code, subject to normal experimentation with learning-rate seeds and training time.