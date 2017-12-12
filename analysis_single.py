import numpy as np
import input_quickdraw
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde

single = input_quickdraw.Single(['diamond'], 5000)
X = single.train_data
Y = single.train_label

vae = VAE(784, [500], 2)
vae.train(X, batch_size=200, num_epochs=100, rerun=False, model_filename='diamond_only')

print('Encoding...')

latent_z = tuple()
for i in range(3):
    latent_z = latent_z + (vae.encode(X), )
latent_z = np.concatenate(latent_z)

print('Binning...')

nbins = 100
x, y = latent_z.T
k = kde.gaussian_kde(latent_z.T)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.clf()
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Reds')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.axes().set_aspect('equal')
plt.show()

image_rows = tuple()
for yr in np.linspace(-3, 3, 8):
    image_row = tuple()
    for xr in np.linspace(-3, 3, 8):
	x_reconstructed = vae.decode(np.array([[xr,yr]]))
        image_row = image_row + (x_reconstructed[0].reshape(28,28),)
    image_rows = image_rows + (np.hstack(image_row),)
image = np.vstack(image_rows)
plt.imshow(image, cmap='Greys')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.show()
