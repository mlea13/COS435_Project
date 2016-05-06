import cv2
import sys
import os
import glob
import classify as alexnet

pathname = sys.argv[1]

top_dirs = glob.glob(pathname + "*/")
for top_dir in top_dirs:
    imgs = glob.glob(top_dir + "*.jpg")
    dirs = glob.glob(top_dir + "*/")
    f1 = open(top_dir + 'classification.txt', 'w')
    
    #classify each image
    for im_file in imgs:
        im = cv2.imread(im_file)
        #im_size = cv2.resize(im, (256, 256))
        alexnet.predict(im_file, im, f1, "jpg")
    f1.close()
        
    #run through each dir
    #for a_dir in dirs:
     #   distorted_imgs = glob.glob(a_dir + "*.png")
     #   f2 = open(a_dir + 'classification.txt', 'w')
     #   
     #   #classify each image
     #   for dist_file in distorted_imgs:
     #       dist = cv2.imread(dist_file)
     #       #dist_size = cv2.resize(dist, (256, 256))
     #       alexnet.predict(dist_file, dist, f2, "png")
     #   f2.close()
