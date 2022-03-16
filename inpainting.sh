# Run inpainting for an image.

# monar_256: Name of the ground-truth image in /images/inpainting/true
# p: p percent of the pixels is missing in the corrupted image.
# num_iter: Number of iterations of DIP.
# check: If this flag is set, checks whether the output files are already exists.
# outputs are saved under the directory /benchmark/inpainting
python inpainting.py monar_256 --p 50 --num_iter 9500 --check
