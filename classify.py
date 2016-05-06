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


def predict(input_file, output_file):
    pycaffe_dir = "/home/cos326/caffe/python/"

    #model_def = "/home/cos326/COS435_Project/alex_net/deploy.prototxt"
    #pretrained_model = "/home/cos326/COS435_Project/alex_net/bvlc_alexnet.caffemodel"
    model_def = "/home/cos326/COS435_Project/google_net/deploy.prototxt"
    pretrained_model = "/home/cos326/COS435_Project/google_net/bvlc_googlenet.caffemodel"
    
    mean = np.load("/home/cos326/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy")
    labels = np.loadtxt("/home/cos326/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
    
    raw_scale = 255.0
    channel_swap = [2,1,0]
    ext = 'jpg'
    image_dims = [256, 256]
    input_scale = 1.0
    
    caffe.set_mode_cpu()
    
    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    inputs = [input_file] #maybe do directory..#

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs, True)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    #print("Saving results into %s" % output_file)
    #np.save(output_file, predictions)
    top_3 = predictions[0].argsort()[-3:][::-1]
    print top_3
    top_3_labels = labels[top_3]
    print top_3_labels
    print predictions[0][top_3]
