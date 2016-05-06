import cv2
import sys
import os
import classify as alexnet

filename = sys.argv[1]

im = cv2.imread(filename)
im_size = cv2.resize(im, (256, 256))

alexnet.predict(im_size, "foo")
