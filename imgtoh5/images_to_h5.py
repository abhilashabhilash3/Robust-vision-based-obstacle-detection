import os
import h5py
import argparse
import numpy as np
import glob
import datetime as dt
import cv2

SOURCE_IMAGES = os.path.abspath("/home/abhilash/Documents/imgtoh5/SampleImages/")
print "[INFO] images paths reading"
images = os.path.join(SOURCE_IMAGES, "*.jpg")
data_order = 'tf' 

if data_order == 'th':
    train_shape = (len(images), 3, 640, 480)
else:
    train_shape = (len(images), 640, 480, 3)
print "[INFO] h5py file created"

hf=h5py.File('data.h5', 'w')

hf.create_dataset("times",
                  shape=train_shape,
                  maxshape=train_shape,
                  compression="gzip",
                  compression_opts=9)

print "[INFO] read and size images"
for i,addr in enumerate(images):

    s=dt.datetime.now()
    img = cv2.imread(images[i])
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

    hf["times"][i, ...] = img[None]
    e=dt.datetime.now()
    print "[INFO] image",str(i),"is saved time:", e-s, "second"

hf.close()
