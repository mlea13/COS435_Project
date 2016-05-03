import sys
import cv2
import numpy as np

def change_hsv(column, img, amt):
    temp = img.copy()
    temp = temp.astype(float)
    temp[:, :, column] += amt
    low = temp < 0
    high = temp > 255
    temp[low] = 0
    temp[high] = 255
    return cv2.cvtColor(temp.astype(np.uint8), cv2.COLOR_HSV2RGB)

def warp_color(a, w, img):
    temp = img.copy()
    A = temp.shape[0] / a
    W = w / temp.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x*W)
    
    for i in range(temp.shape[1]):
        temp[:,i] = np.roll(img[:,i], int(shift(i)))
    return temp

def warp(a, w, img):
    b,g,r = cv2.split(img)
    b_warp = warp_color(a, w, b)
    g_warp = warp_color(a, w, g)
    r_warp = warp_color(a, w, r)   
    return cv2.merge((b_warp,g_warp,r_warp))

def noise(mean, var, sig, img):
    temp = img.copy()
    height, width, _ = img.shape
    sigma = var**sig
    gauss = np.random.normal(mean, sigma, (height, width, 3))
    gauss = gauss.reshape(height, width, 3)
    noise = temp + gauss
    return noise

def distort_image(pathname, filename):
    im = cv2.imread(filename)
    height, width, _ = im.shape
    
    #smooth
    blur1 = cv2.blur(im,(25,25))
    blur2 = cv2.blur(im,(35,35))
    blur3 = cv2.blur(im,(45,45))
    
    cv2.imwrite(pathname + "_b1.png", blur1)
    cv2.imwrite(pathname + "_b2.png", blur2)
    cv2.imwrite(pathname + "_b3.png", blur3)
    
    #color
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    
    h1 = change_hsv(0, hsv, -70)
    h2 = change_hsv(0, hsv, -50)
    h3 = change_hsv(0, hsv, 50)
    h4 = change_hsv(0, hsv, 70)
    
    cv2.imwrite(pathname + "_h1.png", h1)
    cv2.imwrite(pathname + "_h2.png", h2)
    cv2.imwrite(pathname + "_h3.png", h3)
    cv2.imwrite(pathname + "_h4.png", h4)
    
    #warp
    w1 = warp(3.0, 2.0, im)
    w2 = warp(6.0, 4.0, im)
    w3 = warp(12.0, 8.0, im)
    
    cv2.imwrite(pathname + "_w1.png",  w1)
    cv2.imwrite(pathname + "_w2.png",  w2)
    cv2.imwrite(pathname + "_w3.png",  w3)
    
    #noise
    n1 = noise(0, 10, 1, im)
    n2 = noise(0, 30, 1, im)
    n3 = noise(0, 70, 1, im)
    
    cv2.imwrite(pathname + "_n1.png", n1)
    cv2.imwrite(pathname + "_n2.png", n2)
    cv2.imwrite(pathname + "_n3.png", n3)
