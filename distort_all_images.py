import glob
import sys
import os
import change_image as ci

pathname = sys.argv[1]

#cycle through all images in all directories
#top_dirs = glob.glob(pathname+ "*/")
#for top_dir in top_dirs:
dirs = glob.glob(pathname + "*/")
for a_dir in dirs:
    imgs = glob.glob(a_dir+ "*.jpg")
    for im in imgs:
        im_number = im.split("/")[-1].split(".")[0]
        new_path = a_dir + im_number + "/" + im_number
        print new_path
        ci.distort_image(new_path, im)
