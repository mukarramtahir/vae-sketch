import sys
import numpy as np
import input_cartoons
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde
from sklearn.manifold import TSNE

z_sub = {'bat': 1, 'bac': 1, 'bbt': 1, 'bbc': 1, 'but': 1, 'buc': 1, 'fat': 2, 'fac': 2, 'fbt': 2, 'fbc': 2, 'fut': 2, 'fuc': 2}
z_obj = {'bat': 1, 'bac': 2, 'bbt': 1, 'bbc': 2, 'but': 1, 'buc': 2, 'fat': 1, 'fac': 2, 'fbt': 1, 'fbc': 2, 'fut': 1, 'fuc': 2}
z_loc = {'bat': 3, 'bac': 3, 'bbt': 2, 'bbc': 2, 'but': 1, 'buc': 1, 'fat': 3, 'fac': 3, 'fbt': 2, 'fbc': 2, 'fut': 1, 'fuc': 1}

single = input_cartoons.Cartoon(['bac', 'bat', 'bbc', 'bbt', 'buc', 'but', 'fac', 'fat', 'fbc', 'fbt', 'fuc', 'fut'])
X = single.train_data
Y = single.train_label
C = single.train_captions
print single.color_scheme

latent_dim = 2
vae = VAE(784, [500], latent_dim)
vae.train(X, batch_size=20, num_epochs=100, rerun=False, model_filename='composite_l_%d' % latent_dim)

if latent_dim == 2:
    latent_z = vae.encode(X)
    
    plt.clf()
    latent_y = [z_sub[y] for y in Y]
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_y, cmap='Set1')
    plt.axes().set_aspect('equal')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()
    
    plt.clf()
    latent_y = [z_obj[y] for y in Y]
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_y, cmap='Set1')
    plt.axes().set_aspect('equal')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    plt.clf()
    latent_y = [z_loc[y] for y in Y]
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_y, cmap='Set1')
    plt.axes().set_aspect('equal')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

    image_rows = tuple()
    for yr in np.linspace(-5, 5, 8):
	image_row = tuple()
	for xr in np.linspace(-5, 5, 8):
	    x_reconstructed = vae.decode(np.array([[xr,yr]]))
	    image_row = image_row + (x_reconstructed[0].reshape(28,28),)
	image_rows = image_rows + (np.hstack(image_row),)
    image = np.vstack(image_rows)
    plt.clf()
    plt.imshow(image, cmap='Greys')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.show()

