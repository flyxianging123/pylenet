#!/usr/bin/env python
#coding:utf-8
import numpy as np
import struct
import cv2

import matplotlib.pyplot as plt
from ifunction import Batch_Normalization

CHAR = "0123456789"


def loadImageSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    return imgs


def loadLabelSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    return labels


class mnist:
    def __init__(self, batch_size):
        self.train_images_in = open("/home/xiangrong/lx/pycnn/MNIST_data/train-images-idx3-ubyte", 'rb')
        self.train_labels_in = open("/home/xiangrong/lx/pycnn/MNIST_data/train-labels-idx1-ubyte", 'rb')
        self.test_images_in = open("/home/xiangrong/lx/pycnn/MNIST_data/t10k-images-idx3-ubyte", 'rb')
        self.test_labels_in = open("/home/xiangrong/lx/pycnn/MNIST_data/t10k-labels-idx1-ubyte", 'rb')
        self.batch_size = batch_size
        self.train_image = loadImageSet(self.train_images_in)                            # [60000, 1, 784]
        self.train_labels = loadLabelSet(self.train_labels_in)                           # [60000, 1]
        self.test_images = loadImageSet(self.test_images_in)                             # [10000, 1, 784]
        self.test_labels = loadLabelSet(self.test_labels_in)                             # [10000, 1]
        self.data = {"train": self.train_image, "test": self.test_images}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}

    def get_mini_bath(self, data_name="train"):
        if self.indexes[data_name]*self.batch_size > self.data[data_name].shape[0]:
            self.indexes[data_name] = 0
        batch_data = self.data[data_name][self.indexes[data_name]*self.batch_size:(self.indexes[data_name]+1)*self.batch_size, :, :]
        batch_label = self.label[data_name][self.indexes[data_name]*self.batch_size:(self.indexes[data_name]+1)*self.batch_size, :]
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, len(CHAR)))
        for kk in range(self.batch_size):
            y[kk, CHAR.index(str(int(batch_label[kk])))] = 1.0
        x = Batch_Normalization(batch_data)
        return x, y
load_da
