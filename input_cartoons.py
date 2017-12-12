from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from os import listdir

import numpy as np
import matplotlib.pyplot as plt

class Cartoon:
    def __init__(self, class_names, path='cartoon_data'):
        self.caption_dict = {}
        with open(path + '/captions.txt', 'r') as f:
            for line in f:
                line = line.split()
                self.caption_dict[line[0]] = " ".join(line[1:])

        self.train_data = []
        self.train_label = []
        self.train_captions = []

        self.color_scheme = {}
        for i, cn in enumerate(class_names):
            self.color_scheme[cn] = i + 1

        for f in listdir(path):
            if f[:3] in class_names:
                image = Image.open('cartoon_data/%s' % f)
                image = image.resize((28, 28), Image.ANTIALIAS) 
                image = np.asarray(image)
                image = image / 255.
                image = np.ravel(image)

                self.train_data.append(image.tolist())
                self.train_label.append(f[:3])
                self.train_captions.append(self.caption_dict[f[:-4]])

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
