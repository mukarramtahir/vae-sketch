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


#latent_z = np.random.uniform(low=-6, high=6, size=(10000, 40))
latent_z = vae.encode(X)
x_reconstructed = vae.decode(latent_z)
#for i in range(len(x_reconstructed)):
for n, i in enumerate(np.random.randint(low=0, high=x_reconstructed.shape[0], size=1000)):
    print i
    plt.clf()
    plt.imshow(x_reconstructed[i].reshape(28,28), cmap='Greys')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.box(on=False)
    plt.savefig('/home/mukarram/novel_rec/l_%d_v_%d.pdf' % (latent_dim, n + 1), bbox_inches='tight')
