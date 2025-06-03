# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-checkerboard
**Analysis Date:** 2025-05-25

## Analysis Results

1. Brief paper summary and core claims  
• SliceGAN is a GAN-based “dimensionality–expansion’’ framework that trains on 2-D micrographs and synthesises statistically-consistent 3-D volumes.  
• Key ideas:  
 – “Slice” trick – a 3-D generator produces a cube, all axial slices (3 l of them) are fed to a 2-D discriminator that is never aware of the third dimension.  
 – Uniform–information–density generator.  Convolution-transpose layers are constrained to k = 4, s = 2, p = 2 (or other (k,s,p) that satisfy k mod s = 0 and p ≥ k-s) so that every voxel is influenced by the same number of parameters and edge artefacts disappear; the latent seed has a spatial size 4×4×4 so that the first layer already learns overlapping kernels and, once trained, the generator can be asked for arbitrarily large volumes.  
 – Wasserstein-GAN-GP training, typically with critic:generator updates = 5:1, λGP = 10, batch = 8, image edge = 64 vox.  
• Demonstrations on seven very different microstructures, including an Li-ion NMC cathode where SliceGAN’s synthetic volumes reproduce two-point correlation, triple-phase-boundary density and effective diffusivity as well as a 3-D-to-3-D GAN trained on tomograms.

2. Implementation assessment  
Execution path for “python run_slicegan 1” (training) or “… 0” (generation):  
• run_slicegan.py sets all hyper-parameters, creates output folders, then calls  

 netD, netG = networks.slicegan_rc_nets( … )  
 model.train( … )  (for training) or util.test_img(..) (for inference)

• networks.slicegan_rc_nets builds  
 – Generator: a list of ConvTranspose3d layers followed by an Upsample(…,"trilinear") and a final 3 × 3 × 3 Conv3d (“rcconv”). Soft-max is applied for n-phase data.  
 – Discriminator: a conventional 2-D CNN.  
 Layer parameters are taken from the lists supplied in run_slicegan (gk, gs, gp for the generator; dk, ds, dp for the discriminator).

• Default settings in run_slicegan:  
 gk = [4,4,4,4,4]  
 gs = [3,3,3,3,3]  <-- stride = 3 (paper uses 2)  
 gp = [1,1,1,1,1]  
 Latent tensor noise has shape [batch, 32, 4,4,4] (matches paper).

• model.train implements the slice-based WGAN-GP training loop:  
 – 3 independent discriminators are instantiated; for isotropic problems the same one is reused.  
 – For every critic step the fake cube (64³ by default) is permuted and reshaped into l × batch 2-D images before being scored by the discriminator, matching the “Slice” idea.  
 – Gradient penalty and Wasserstein losses are coded as described in the paper.

• preprocessing.py produces the required 32 × 900 random 2-D crops per orientation (or per colour channel) and one-hot-encodes multi-phase data exactly as in the text.

3. Discrepancies between paper and code  

| # | Observation | Classification | Comment |
|---|-------------|----------------|---------|
|1|Generator strides are set to 3 (gs = 3) and padding = 1 by default, so k mod s ≠ 0 and the uniform-information-density rule claimed essential in §4 of the paper is violated. The code therefore falls back to an Upsample + 3 × 3 conv (“rcconv”) that is **not** mentioned in the paper.|Critical*|With the parameters shipped in run_slicegan the published edge-artefact avoidance strategy is not reproduced. Using networks.slicegan_nets instead of slicegan_rc_nets **or** changing gs to 2 and gp to 2 would match the paper.|
|2|The “rc” generator introduces an extra trilinear Upsample while the paper states that all layers are transpose-convolutions.|Minor|May change visible texture but does not contradict the general slice-GAN idea.|
|3|k-sizing of Upsample: size is calculated with `x.shape[3]` twice (probable typo, should be shape[4] for the last dimension).|Minor|Works because for cubic volumes shape[3]==shape[4]; still a code–paper mismatch.|
|4|Gradient-penalty, critic_iters, λGP, batch sizes (8/8) match the text values, but number of epochs (100) and learning rates (1e-4) are not explicitly stated in the paper.|Cosmetic|Typical hyper-parameter disclosure issue.|
|5|Implementation always instantiates three discriminators even for isotropic data (paper suggests one discriminator is sufficient). The extra two are immediately aliased to the first.|Cosmetic|No effect on results, just inefficiency.|

\*Whether #1 is “critical’’ depends on the user: if the default rc-network still produces satisfactory volumes the core claim (2-D → 3-D via slices) holds, but the specific edge-quality argument made in the paper cannot be tested unless the parameters are adjusted.

4. Overall reproducibility conclusion  
The code base clearly implements the central “slice discriminator + 3-D generator” idea and uses the training scheme described in the paper.  A reader can reproduce the headline claim—training on a single 2-D section and generating plausible 3-D volumes—without major difficulty.

However, the default network defined in run_slicegan deviates from the architecture stressed in the manuscript (different stride, added up-sampling layer).  Anyone wishing to replicate the edge-uniformity study (§4) or the exact statistics reported for the NMC cathode must modify gk/gs/gp (or switch to `slicegan_nets`) to restore the k = 4, s = 2, p = 2 stack.

In summary, the work is largely reproducible, but one crucial architectural parameter set is different in the released script and needs to be corrected to match the publication’s generator specification.