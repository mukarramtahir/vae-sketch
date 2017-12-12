import numpy as np
import os

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from os import listdir

'''
for filename in os.listdir(path):
	os.rename(os.path.join(path, filename), os.path.join(path, filename.replace(" ", "_")))
'''

class Single:
    def __init__(self, class_names, samples_per_class, path='quickdraw_data'):
	# Define color scheme
	self.color_scheme = {}
	for i, l in enumerate(class_names):
	    self.color_scheme[l] = i + 1

	train_label = []
	train_data = np.zeros((0,784))
	for label in class_names:
	    raw_data = np.load(os.path.join(path, label + '.npy'))[0:samples_per_class,:]
	    train_data = np.append(train_data, raw_data/255.0, axis = 0)
	    for j in range(samples_per_class):
		train_label.append(label)
	
	self.train_label = np.array(train_label)
	self.train_data = np.array(train_data)


'''
plt.figure()
(d,l) = next_batch()
image = np.reshape(d[1,:],(28,28))
print(l)
plt.imshow(image,cmap = 'Greys')
plt.savefig('test_image')
'''
