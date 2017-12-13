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

#xr = float(sys.argv[1])
#yr = float(sys.argv[2])
a = float(sys.argv[1])
b = float(sys.argv[2])
#c = float(sys.argv[3])
#d = float(sys.argv[4])
#e = float(sys.argv[5])
#x_reconstructed = vae.decode(np.array([[a,b,c,d,e]]))
x_reconstructed = vae.decode(np.array([[a,b]]))
plt.imshow(x_reconstructed[0].reshape(28,28), cmap='Greys')
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.box(on=False)
plt.savefig('../final_figures/%s' % sys.argv[3], bbox_inches='tight')

