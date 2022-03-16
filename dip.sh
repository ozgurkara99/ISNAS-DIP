# Run dip. [1]

# f16_512_rgb: Name of the ground-truth image in /images/denoising/true
# sigma: Std of noise.
# num_iter: Number of iterations of DIP.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python dip.py f16_512_rgb --sigma 25 --num_iter 1200 --save_out_at 1000,1100,1200 --check

# monar_256: Name of the ground-truth image in /images/inpainting/true
# p: p percent of the pixels is missing in the corrupted image.
# num_iter: Number of iterations of DIP.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python dip.py monar_256 --p 50 --num_iter 9500 --save_out_at 8500,8700,9000 --check

# woman_rgb: Name of the ground-truth image in /images/sr/true
# zoom: Scaling factor for the low-resolution image.
# num_iter: Number of iterations of DIP.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python dip.py woman_rgb --zoom 4 --num_iter 4500 --save_out_at 1000,1100,1200 --num_models 5000 --check


[1] https://dmitryulyanov.github.io/deep_image_prior
