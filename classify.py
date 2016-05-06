#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import glob
import time

import caffe


def predict(input_file, output_file, ext):
    pycaffe_dir = "/home/cos326/caffe/python/"
    model_def = "/home/cos326/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
    pretrained_model = "/home/cos326/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
    images_dim = '256,256'
    mean_file ="/home/cos326/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"
    raw_scale = 255.0
    channel_swap ='2,1,0'
    input_scale = None
    center_only = None
    
    labels = np.loadtxt("/home/cos326/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')

    image_dims = [int(s) for s in images_dim.split(',')]
    mean = np.load(mean_file)
    channel_swap = [int(s) for s in channel_swap.split(',')]
    caffe.set_mode_cpu()

    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.expanduser(input_file)
    if input_file.endswith('npy'):
        print("Loading file: %s" % input_file)
        inputs = np.load(input_file)
    elif os.path.isdir(input_file):
        print("Loading folder: %s" % input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(input_file + '/*.' + ext)]
    else:
        print("Loading file: %s" % input_file)
        inputs = [caffe.io.load_image(input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not center_only)
    print("Done in %.2f s." % (time.time() - start))
                    

    # Save
    #print("Saving results into %s" % output_file)
    #np.save(output_file, predictions)
    top_3 = predictions[0].argsort()[-3:][::-1]
    top_3_labels = labels[top_3]

    output_file.write("########\n")
    output_file.write(input_file)
    output_file.write("\n")
    output_file.write(top_3_labels)
    output_file.write("\n")
    #output_file.write(predictions[0][top_3])
    #output_file.write("\n")
