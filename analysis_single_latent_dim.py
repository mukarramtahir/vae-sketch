import sys
import numpy as np
import input_quickdraw
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde
from sklearn.manifold import TSNE

single = input_quickdraw.Single(['barn'], 5000)
X = single.train_data
Y = single.train_label

latent_dim = int(sys.argv[1])
vae = VAE(784, [500], latent_dim)
vae.train(X, batch_size=200, num_epochs=100, rerun=False, model_filename='barn_l_%d' % latent_dim)

latent_z = vae.encode(X)

x_reconstructed = vae.decode(latent_z)
for n, i in enumerate(np.random.randint(low=0, high=x_reconstructed.shape[0], size=20)):
    print i
    plt.clf()
    plt.imshow(x_reconstructed[i].reshape(28,28), cmap='Greys')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.box(on=False)
    plt.savefig('/home/mukarram/barn_l_%d_v_%d.pdf' % (latent_dim, n + 1), bbox_inches='tight')


'''
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
'''
