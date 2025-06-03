# Paper-Code Consistency Analysis (OpenAI)

**Paper:** SliceGAN-paper  
**Code Archive:** SliceGAN-master-softmax
**Analysis Date:** 2025-05-25

## Analysis Results

1. Paper summary and core claims  
• Paper presents SliceGAN, a GAN architecture that can be trained with one (isotropic) or three (anisotropic) 2-D micrographs and afterwards generates arbitrarily large 3-D micro-structures.  
• Key methodological points:  
  – Generator is a fully-convolutional 3-D de-convolution network (5 layers, kernel 4, stride 2, padding 2 except final layer p = 3) that receives a 64 × 4 × 4 × 4 latent tensor.  
  – Training follows WGAN-GP; every generated cube is sliced in the three orthogonal directions and all 64 slices per axis are shown to a 2-D discriminator. Recommended batch ratio mG = 2 mD.  
  – For n-phase data the generator output is passed through a channel-wise soft-max so that each voxel stores a phase-probability distribution.  
  – Non-uniform information density in de-convolutions is avoided by the {k=4, s=2, p=2} rule set.  
  – After training the fully-convolutional generator can be asked for volumes of any size.

2. Implementation assessment  
• Entry point run_slicegan.py builds a project, chooses image type, then instantiates networks via slicegan_rc_nets() and runs training through model.train().  
• Training loop (model.py) follows WGAN-GP, uses three separate discriminators, slices the fake cube with tensor permutations and reshaping exactly as described and applies a gradient penalty λ = 10.  
• Generator/Discriminator definitions (networks.py) are parameterised; default parameters in run_slicegan reproduce the “5-layer transpose-conv” design and use k=4, s=2, p = [2,2,2,2,3].  
• Data handling (preprocessing.py) extracts 64×64 slices from tif stacks exactly as in the paper.  
• Utility routines implement post-processing, progress printing, graph logging, and volume export to .tif.

3. Discrepancies  

| # | Description | Paper | Code | Impact | Class |
|---|-------------|-------|------|--------|-------|
|1|Number of filters in early generator layers|512, 256, 128, 64|1024, 512, 128, 32 (run_slicegan defaults)|Affects parameter count / memory, but not the algorithmic idea|Minor|
|2|Final activation for n-phase data|Soft-max over channels|Sigmoid in slicegan_rc_nets (soft-max only in slicegan_nets)|Output channels are independent probabilities instead of a normalized distribution; post-processing still takes arg-max so images are produced, but voxel probabilities no longer sum to 1|Minor|
|3|Additional up-sampling + 3×3×3 Conv (“rcconv”) inserted after last transpose-conv|Not mentioned|Present in slicegan_rc_nets|Extra refinement stage, still convolutional, does not change overall concept|Minor|
|4|Generator/discriminator batch ratio|mG = 2 mD suggested|batch_size = D_batch_size = 8 (1 : 1)|May change convergence speed but not principle|Cosmetic|
|5|Training epochs / learning rates|Not fixed in paper|num_epochs = 100, lr = 1e-4|Typical hyper-parameter choice|Cosmetic|

No discrepancies were found that invalidate the algorithmic description (e.g. slicing strategy, WGAN-GP training, information-density rule set).

4. Overall reproducibility conclusion  
The public code implements the core ideas of SliceGAN: a 3-D transpose-convolution generator trained only with 2-D slices, WGAN-GP optimisation, and the data-slicing trick that bridges the 2-D/3-D mismatch. Architectural and hyper-parameter details differ slightly from the architecture table in the paper, but these deviations are limited to filter counts, an extra refinement convolution, and the use of sigmoid rather than soft-max in one variant. They should not prevent a reader from reproducing the central qualitative and quantitative claims of the paper; at most they may require minor tuning to match exact numerical results. Accordingly, reproducibility can be considered good, with only minor adjustments needed.