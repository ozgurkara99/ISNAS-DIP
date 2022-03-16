# Run random search. Read models from the file models.lst. Calculate the metrics for each of them. Sort the models according to the metrics. Select the top 15 models. Train 15 models using DIP. Select the model whose output is closest to average of the outputs of the 15 models.

# f16_512_rgb: Name of the ground-truth image in /images/denoising/true
# sigma: Std of noise.
# num_iter: Number of iterations of DIP.
# num_models: Number of randomly generated models.
# small: If this flag is set, resized-averaging technique is used.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python random_search.py f16_512_rgb --sigma 25 --num_iter 1200 --num_models 5000 --save_out_at 1000,1100,1200 --small --check

# monar_256: Name of the ground-truth image in /images/inpainting/true
# p: p percent of the pixels is missing in the corrupted image.
# num_iter: Number of iterations of DIP.
# num_models: Number of randomly generated models.
# small: If this flag is set, resized-averaging technique is used.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python random_search.py monar_256 --p 50 --num_iter 9500 --num_models 5000 --save_out_at 8500,8700,9000 --small --check

# woman_rgb: Name of the ground-truth image in /images/sr/true
# zoom: Scaling factor for the low-resolution image.
# num_iter: Number of iterations of DIP.
# num_models: Number of randomly generated models.
# save_out_at: A list of integer to save the outputs at these iterations.
# check: If this flag is set, checks whether the output files are already exists.
python random_search.py woman_rgb --zoom 4 --num_iter 4500 --save_out_at 3000,3500,4000 --num_models 5000 --check
