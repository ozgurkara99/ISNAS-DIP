Install the necessary packages using:

conda create -n isnasdip python=3.8

pip install -r requirements.txt

If any of the packages listed in requirements.txt is failed to installed, install it manually, remove it from the txt file and rerun the above command.


Go to utils/paths.py and change the variable PROJECT_FOLDER to path of the current directory.

The following directories and files are necessary to run the experiments:

CURRENT DIRECTORY
    images
        denoising
            true
                image1.png
                image2.png
        inpainting
            true
                image1.png
                image2.png
        sr
            true
                image1.png
                image2.png
    
    random_search
        models_generated.lst
    
    benchmark
        models.csv


Put the ground-truth images into the corresponding folder for example into images/denoising/true for denoising. Make sure that the images do not have alpha channels. You can use remove_alpha.py script to remove the alpha channels of all the images:
python remove_alpha.py

To run isnasdip experiment see the isnasdip.sh

To run nasdip experiment see the nasdip.sh

To run dip experiment see the dip.sh


