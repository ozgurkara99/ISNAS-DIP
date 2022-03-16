# Run super-resolution for an image.

# woman_rgb: Name of the ground-truth image in /images/sr/true
# zoom: Scaling factor for the low-resolution image.
# num_iter: Number of iterations of DIP.
# check: If this flag is set, checks whether the output files are already exists.
# outputs are saved under the directory /benchmark/sr
python sr.py woman_rgb --p 50 --num_iter 4500 --check
