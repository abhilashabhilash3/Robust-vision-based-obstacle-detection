# -*- coding: utf-8 -*-

import os
from common import logger
import sys
import time
import signal
import yaml

import tensorflow as tf
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from glob import glob

from scipy.ndimage.filters import correlate1d, _gaussian_kernel1d, _ni_support

def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            sd = float(sigma)
            # make the radius of the filter equal to truncate standard deviations
            lw = int(truncate * sd + 0.5)
            # Since we are calling correlate, not convolve, revert the kernel
            weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

            if input.ndim == 3 and axis == 0:
                weights[weights.size // 2 + 1:] = 0

            correlate1d(input, weights, axis, output, mode, cval, 0)
            input = output
    else:
        output[...] = input[...]
    return output

"""
def median_filter(input, x, y, gridSize):    #x and y are nested loops, that run over the entire image.
    medianList = []
    row, col = input.shape
    k = int(gridSize/2)
    for i in range(gridSize-1):
        for j in range(gridSize-1):
            if (i+x-k)<0 or (j+y-k)<0 or (i+x-k)>row or (j+y-k)>col:
                break
            medianList.append(input[(i+x-k),(j+y-k)])
    medianList.sort()
    length = len(medianList)
    if length%2 != 0:
        output = float(medianList[length/2])
    output = float((medianList[int((length-1)/2)] + medianList[int(length/2)]) / 2.0);
    return output
"""
###############
#  Images IO  #
###############

def load_dataset(files):
    if isinstance(files, basestring):
        files = [files]
        
    # Check parameters
    if not files or len(files) < 1 or files[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % files)

    # Load dataset
    if files[0].endswith(".tfrecord"):
        dataset = load_tfrecords(files)

        # Get number of examples in dataset
        total = sum(1 for record in tqdm(dataset, desc="Loading dataset", file=sys.stderr))

    elif files[0].endswith(".jpg"):
        dataset = load_jpgs(files)

        # Expand wildcards
        files_expanded = []
        for s in files:
            files_expanded += glob(s)
        total = len(list(set(files_expanded))) # Remove duplicates
    else:
        raise ValueError("Supported file types are *.tfrecord and *.jpg")

    return dataset, total

def load_tfrecords(filenames, batch_size=64):
    """Loads a set of TFRecord files
    Args:
        filenames (str / str[]): TFRecord file(s) extracted
                                 by rosbag_to_tfrecord

    Returns:
        tf.data.MapDataset
    """
    if not filenames or len(filenames) < 1 or filenames[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % filenames)
    
    raw_dataset = tf.data.TFRecordDataset(filenames)

    # Create a dictionary describing the features.
    feature_description = {
        "metadata/time"     : tf.io.FixedLenFeature([], tf.int64), # TODO: Change to int64
        "image/height"      : tf.io.FixedLenFeature([], tf.int64),
        "image/width"       : tf.io.FixedLenFeature([], tf.int64),
        "image/channels"    : tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace"  : tf.io.FixedLenFeature([], tf.string),
        "image/format"      : tf.io.FixedLenFeature([], tf.string),
        "image/encoded"     : tf.io.FixedLenFeature([], tf.string)
    }
    
    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_example(example_proto, feature_description)

    def _decode_function(example):
        image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
        time = example["metadata/time"]
        return image, time

    return raw_dataset.batch(batch_size) \
                      .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .unbatch() \
                      .map(_decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .prefetch(tf.data.experimental.AUTOTUNE)

def load_jpgs(filenames):
    """Loads a set of TFRecord files
    Args:
        filenames (str / str[]): JPEG file(s) extracted
                                 by rosbag_to_images

    Returns:
        tf.data.MapDataset
    """
    if not filenames or len(filenames) < 1 or filenames[0] == "":
        raise ValueError("Please specify at least one filename (%s)" % filenames)
    
    raw_dataset = tf.data.Dataset.list_files(filenames, shuffle=False)

    def _decode_function(file_path):
        # Load and decode image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Get the time from the file path
        time = tf.strings.split(file_path, '/')[-1]
        time = tf.strings.split(time, ".")[0]
        time = tf.strings.to_number(time, out_type=tf.dtypes.int64)
        return image, time

    return raw_dataset.map(_decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                      .prefetch(tf.data.experimental.AUTOTUNE)

#################
# Output helper #
#################

def format_duration(t):
    """Format duration in seconds to a nice string (e.g. "1h 5m 20s")
    Args:
        t (int / float): Duration in seconds

    Returns:
        str
    """
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    output = "%is" % seconds
    
    if (minutes > 0):
        output = "%im %s" % (minutes, output)
        
    if (hours > 0):
        output = "%ih %s" % (hours, output)

    return output

#################
# Computer Info #
#################

import cpuinfo
import GPUtil
from psutil import virtual_memory

def getComputerInfo():

    # Get CPU info
    # cpu = cpuinfo.get_cpu_info()

    result_dict = {
        # "Python version": cpu["python_version"],
        "TensorFlow version": tf.version.VERSION,
        # "CPU Description": cpu["brand"],
        # "CPU Clock speed (advertised)": cpu["hz_advertised"],
        # "CPU Clock speed (actual)": cpu["hz_actual"],
        # "CPU Architecture": cpu["arch"]
    }

    # Get GPU info
    gpus_tf = tf.config.experimental.list_physical_devices("GPU")
    
    result_dict["Number of GPUs (tf)"] = len(gpus_tf)

    gpus = GPUtil.getGPUs()
    gpus_available = GPUtil.getAvailability(gpus)
    for i, gpu in enumerate(gpus):
        result_dict["GPU %i" % gpu.id] = gpu.name
        result_dict["GPU %i (driver)" % gpu.id] = gpu.driver
        result_dict["GPU %i (memory total)" % gpu.id] = gpu.memoryTotal
        result_dict["GPU %i (memory free)" % gpu.id] = gpu.memoryFree
        result_dict["GPU %i (available?)" % gpu.id] = gpus_available[i]

    # Get RAM info
    mem = virtual_memory()

    result_dict["RAM (total)"] = mem.total
    result_dict["RAM (available)"] = mem.available

    return result_dict

#################
#     Misc      #
#################

# https://gist.github.com/nonZero/2907502
class GracefulInterruptHandler(object):

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):

        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):

        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True

class _DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
