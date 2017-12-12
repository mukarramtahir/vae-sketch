import input_quickdraw
import tflearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import kde
from sklearn.manifold import TSNE

#model_filename = 'model.tfl'
model_filename = None

X = input_quickdraw.train_data
Y = input_quickdraw.train_label

input_dim = X.shape[1]
hidden_dim = [500, 500]
latent_dim = 5

# Encoder: Input layer
encoder = tflearn.input_data(shape=[None, input_dim], name='input')

# Encoder: Hidden layer(s)
for i, d in enumerate(hidden_dim):
    encoder = tflearn.fully_connected(encoder, d, activation='sigmoid', scope='hidden_layer_%d' % (i + 1))
    encoder = tflearn.batch_normalization(encoder)
    encoder = tflearn.dropout(encoder, 0.4)

# Map to latent space
z_mean = tflearn.fully_connected(encoder, latent_dim, scope='z_mean')
z_std = tflearn.fully_connected(encoder, latent_dim, scope='z_std')

# Sample from latent space
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
z = z_mean + tf.exp(z_std / 2.) * eps

# Decoder: Hidden layers(s)
for i, d in enumerate(reversed(hidden_dim)):
    if i == 0:
        decoder = tflearn.fully_connected(z, d, activation='relu', scope='decoder_layer_%d' % (i + 1))
        decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1))
        decoder = tflearn.dropout(decoder, 0.4)
    else:
        decoder = tflearn.fully_connected(decoder, d, activation='relu', scope='decoder_layer_%d' % (i + 1))
        decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1))
        decoder = tflearn.dropout(decoder, 0.4)

decoder = tflearn.fully_connected(decoder, input_dim, activation='sigmoid', scope='decoder_output')

def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
			+ (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)

    # KL-divergence loss
    kl_div_loss = 1 + (2 * z_std) - tf.square(z_mean) - tf.exp(2 * z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

net = tflearn.regression(decoder, optimizer='rmsprop', learning_rate=0.001, loss=vae_loss, metric=None, name='target')

training_model = tflearn.DNN(net, tensorboard_verbose=0)

if model_filename is not None:
    training_model.load(model_filename)
else:
    training_model.fit({'input': X}, {'target': X}, n_epoch=50, validation_set=(X, X), batch_size=500, run_id='vae')
    training_model.save('fish.tfl')

plt.clf()
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
idx = idx[:100]
X_test = X[idx]
X_out = training_model.predict({'input': X_test})
for i in range(100):
    plt.subplot(10,10, i + 1)
    plt.imshow(X_out[i].reshape(28,28), cmap='Greys')
plt.show()

encoder_model = tflearn.DNN(z, session=training_model.session)
latent_z = encoder_model.predict({'input': X})
latent_z = latent_z[:2000, :]
viz_z = TSNE(n_components=2).fit_transform(latent_z)
print(viz_z)
plt.clf()
plt.scatter(viz_z[:, 0], viz_z[:, 1])
plt.show()

exit()
"""
Visualize latent space
"""
latent_z = tuple()
for i in range(20):
    plt.clf()
    encoder_model = tflearn.DNN(z, session=training_model.session)
    latent_z = latent_z + (encoder_model.predict({'input': X}), )
#latent_colors = [input_quickdraw.color_scheme[i] for i in Y]
latent_z = np.concatenate(latent_z)
#plt.scatter(latent_z[:, 0], latent_z[:, 1])
#plt.show()

print('binning')

nbins = 100
x, y = latent_z.T
k = kde.gaussian_kde(latent_z.T)
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.subplot(1,2,1)
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='Reds')

input_noise = tflearn.input_data(shape=[None, latent_dim], name='input_noise')
for i, d in enumerate(reversed(hidden_dim)):
    if i == 0:
        decoder = tflearn.fully_connected(input_noise, d, activation='relu', scope='decoder_layer_%d' % (i + 1), reuse=True)
        decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1), reuse=True)
        decoder = tflearn.dropout(decoder, 0.4)
    else:
        decoder = tflearn.fully_connected(decoder, d, activation='relu', scope='decoder_layer_%d' % (i + 1), reuse=True)
        decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1), reuse=True)
        decoder = tflearn.dropout(decoder, 0.4)

decoder = tflearn.fully_connected(decoder, input_dim, activation='sigmoid', scope='decoder_output', reuse=True)
generator_model = tflearn.DNN(decoder, session=training_model.session)

n = 1
image_rows = tuple()
for yr in np.linspace(2.0, -0.5, 30):
    image_row = tuple()
    for xr in np.linspace(-3.5, 1, 30):
        x_reconstructed = generator_model.predict({'input_noise': np.array([[xr,yr]])})
        image_row = image_row + (x_reconstructed[0].reshape(28,28),)
        n += 1
    image_rows = image_rows + (np.hstack(image_row),)
image = np.vstack(image_rows)
plt.subplot(1,2,2)
plt.imshow(image, cmap='Greys')
plt.show()

exit()

while True:
    input_str = input('Enter latent values z1 z2:')
    xr = float(input_str.split()[0])
    yr = float(input_str.split()[1])
    x_reconstructed = generator_model.predict({'input_noise': np.array([[xr,yr]])})
    plt.imshow(x_reconstructed[0].reshape(28,28), cmap='Greys')
    plt.show()
