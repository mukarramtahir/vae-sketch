from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from os import listdir

import numpy as np
import matplotlib.pyplot as plt

class Cartoon:
    def __init__(self, class_names, path='cartoon_data'):
        self.train_data = []
        self.train_label = []

        for f in listdir(path):
            if f[:3] in class_names:
                image = Image.open('cartoon_data/%s' % f)
                image = image.resize((28, 28), Image.ANTIALIAS) 
                image = np.asarray(image)
                image = image / 255.
                image = np.ravel(image)

                self.train_data.append(image.tolist())
                self.train_label.append(f[:3])

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
