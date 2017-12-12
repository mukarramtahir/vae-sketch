from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from os import listdir

import numpy as np
import matplotlib.pyplot as plt

train = []
labels = []

for f in listdir('cartoon_data'):
    #image = imread('cartoon_data/%s' % f, as_grey=True)
    #image = resize(image, (28, 28), anti_aliasing=True)
    #if f[2] == 't':
    image = Image.open('cartoon_data/%s' % f)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.
    #image = np.abs(image - 1)
    image = np.ravel(image)
    train.append(image.tolist())
    labels.append(f[:3])

n_samples = len(train)

def next_batch(num):
    idx = np.arange(0 , len(train))
    np.random.shuffle(idx)
    idx = idx[:num]
    shuffled_train = [train[i] for i in idx]
    shuffled_labels = [labels[i] for i in idx]
    return np.array(shuffled_train), shuffled_labels
