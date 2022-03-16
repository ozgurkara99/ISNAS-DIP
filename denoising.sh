# Run denoising for an image.

# f16_512_rgb: Name of the ground-truth image in /images/denoising/true
# sigma: Std of noise.
# num_iter: Number of iterations of DIP.
# check: If this flag is set, checks whether the output files are already exists.
# outputs are saved under the directory /benchmark/denoising
python denoising.py f16_512_rgb --sigma 25 --num_iter 1200 --check
