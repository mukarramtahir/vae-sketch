import numpy as np
import input_quickdraw
import matplotlib.pyplot as plt

from vae import VAE
from scipy.stats import kde

single = input_quickdraw.Single(['alarm_clock'], 5000)
X = single.train_data
Y = single.train_label

vae = VAE(784, [500], 10)
vae.train(X, batch_size=200, num_epochs=100, rerun=False, model_filename='alarm_clock_schematic')

print('Encoding...')

X = X[:1, :]
plt.imshow(X[0].reshape(28,28), cmap='Greys')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.show()

latent_z = vae.encode(X)
x_reconstructed = vae.decode(latent_z)
plt.imshow(x_reconstructed[0].reshape(28,28), cmap='Greys')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.show()
