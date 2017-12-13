import sys
import numpy as np
import input_cartoons
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde
from sklearn.manifold import TSNE

single = input_cartoons.Cartoon(['bac', 'bat', 'bbc', 'bbt', 'buc', 'but', 'fac', 'fat', 'fbc', 'fbt', 'fuc', 'fut'])
X = single.train_data
Y = single.train_label
C = single.train_captions
print single.color_scheme

latent_dim = int(sys.argv[1])
vae = VAE(784, [500], latent_dim)
vae.train(X, batch_size=20, num_epochs=100, rerun=False, model_filename='composite_l_%d' % latent_dim)

latent_z = vae.encode(X)
with open('latent_space_%d_dim_abbrev.txt' % latent_dim, 'w') as lf:
    for i, lz in enumerate(latent_z):
        if Y[i] in ['bat', 'bbt', 'bac', 'bbc']:
            for z in lz:
                lf.write('%f ' % z)
            lf.write('\n')

with open('captions_%d_dim_abbrev.txt' % latent_dim, 'w') as lf:
    for i, _ in enumerate(latent_z):
        if Y[i] in ['bat', 'bbt', 'bac', 'bbc']:
            lf.write('%s\n' % C[i])

if latent_dim == 2:
    latent_z = vae.encode(X)
    latent_y = [single.color_scheme[y] for y in Y]
    
    plt.clf()
    plt.scatter(latent_z[:, 0], latent_z[:, 1], c=latent_y, s=6, cmap='rainbow', edgecolors='k', linewidth=0.25)
    plt.axes().set_aspect('equal')
    plt.show()
    
    
    image_rows = tuple()
    for yr in np.linspace(-4, 4, 10):
	image_row = tuple()
	for xr in np.linspace(-4, 4, 10):
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
