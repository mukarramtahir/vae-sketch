import sys
import numpy as np
import input_quickdraw
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde
from sklearn.manifold import TSNE

single = input_quickdraw.Single(['alarm_clock', 'aircraft_carrier', 'airplane', 'banana', 'broom', 'barn', 'diamond', 'fish', 'lollipop'], 5000)
X = single.train_data
Y = single.train_label
print single.color_scheme

latent_dim = int(sys.argv[1])
vae = VAE(784, [500], latent_dim)
vae.train(X, batch_size=1000, num_epochs=100, rerun=False, model_filename='nine_objects_l_%d' % latent_dim)


if latent_dim == 2:
    latent_z = vae.encode(X)
    latent_y = [single.color_scheme[y] for y in Y]
    
    plt.clf()
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_y, s=4, cmap='rainbow', edgecolors='k', linewidth=0.5)
    plt.axes().set_aspect('equal')
    plt.show()
    
    
    image_rows = tuple()
    for yr in np.linspace(-2.5, 2.5, 10):
	image_row = tuple()
	for xr in np.linspace(-2.5, 2.5, 10):
	    x_reconstructed = vae.decode(np.array([[xr,yr]]))
	    image_row = image_row + (x_reconstructed[0].reshape(28,28),)
	image_rows = image_rows + (np.hstack(image_row),)
    image = np.vstack(image_rows)
    plt.clf()
    plt.imshow(image, cmap='Greys')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.show()

if latent_dim > 2:
    latent_z = vae.encode(X)
    latent_y = [single.color_scheme[y] for y in Y]
    viz_z = TSNE(n_components=2).fit_transform(latent_z)
    plt.clf()
    plt.scatter(viz_z[:, 0], viz_z[:, 1], c=latent_y, s=4, cmap='Set1', edgecolors='k', linewidth=0.1)
    #plt.scatter(viz_z[:, 0], viz_z[:, 1], c=latent_y, s=5, cmap='Set1')
    plt.show()
