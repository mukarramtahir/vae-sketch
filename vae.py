import argparse
import os
import tflearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import kde
from sklearn.manifold import TSNE

class VAE:
    """
    Variational autoencoder
    """
    def __init__(self, input_dim, \
		 hidden_dim, \
		 latent_dim, \
		 encoder_fn='sigmoid', \
		 decoder_fn='relu', \
		 squashing_fn='sigmoid', \
		 dropout_rate=0.4, \
		 learning_rate=0.001):
	"""
	Initialize network
	"""
	self.input_dim = input_dim
	self.hidden_dim = hidden_dim
	self.latent_dim = latent_dim
	self.encoder_fn = encoder_fn
	self.decoder_fn = decoder_fn
	self.squashing_fn = squashing_fn
	self.dropout_rate = dropout_rate
	self.learning_rate = learning_rate

	self._construct_network()

    def _construct_network(self):
	"""
	Construct computation graph
	"""
	self.encoder = tflearn.input_data(shape=[None, self.input_dim], name='input')
	for i, d in enumerate(self.hidden_dim):
	    self.encoder = tflearn.fully_connected(self.encoder, d, activation=self.encoder_fn, scope='hidden_layer_%d' % (i + 1))
	    self.encoder = tflearn.batch_normalization(self.encoder)
	    self.encoder = tflearn.dropout(self.encoder, self.dropout_rate)

	self.z_mean = tflearn.fully_connected(self.encoder, self.latent_dim, scope='z_mean')
	self.z_std = tflearn.fully_connected(self.encoder, self.latent_dim, scope='z_std')

	self.eps = tf.random_normal(tf.shape(self.z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
	self.z = self.z_mean + tf.exp(self.z_std / 2.) * self.eps

	for i, d in enumerate(reversed(self.hidden_dim)):
	    if i == 0:
		self.decoder = tflearn.fully_connected(self.z, d, activation=self.decoder_fn, scope='decoder_layer_%d' % (i + 1))
		self.decoder = tflearn.batch_normalization(self.decoder, scope='decoder_bn_%d' % (i + 1))
		self.decoder = tflearn.dropout(self.decoder, self.dropout_rate)
	    else:
		self.decoder = tflearn.fully_connected(self.decoder, d, activation=self.decoder_fn, scope='decoder_layer_%d' % (i + 1))
		self.decoder = tflearn.batch_normalization(self.decoder, scope='decoder_bn_%d' % (i + 1))
		self.decoder = tflearn.dropout(self.decoder, self.dropout_rate)

	self.decoder = tflearn.fully_connected(self.decoder, self.input_dim, activation=self.squashing_fn, scope='decoder_output')

	self.net = tflearn.regression(self.decoder, optimizer='rmsprop', \
		    learning_rate=self.learning_rate, loss=self._vae_loss, metric=None, name='target')

    def _vae_loss(self, x_reconstructed, x_true):
	"""
	Reconstruction loss and KL divergence
	"""
    	encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
    	    		+ (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    	encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)

    	kl_div_loss = 1 + (2 * self.z_std) - tf.square(self.z_mean) - tf.exp(2 * self.z_std)
    	kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    	return tf.reduce_mean(encode_decode_loss + kl_div_loss)

    def train(self, X, batch_size=100, \
		num_epochs=100, \
		model_filename='auto', \
		model_path='./models', \
		rerun=True):
	"""
	Training session
	"""
	self.training_model = tflearn.DNN(self.net, tensorboard_verbose=0)

	if model_filename == 'auto':
	    model_basename = 'ldim_%s_hdim_%s_encoderfn_%s_decoderfn_%s_squashing_%s' % \
			    (str(self.latent_dim), "_".join(map(str, self.hidden_dim)), \
			    self.encoder_fn, self.decoder_fn, self.squashing_fn)
	    model_filename = model_path + '/' + model_basename + '.tfl'
	else:
	    model_filename = model_path + '/' + model_filename

	if (not os.path.isfile(model_filename + '.index')) or rerun:
	    run_training = True
	else:
	    run_training = False

	if run_training:
	    self.training_model.fit({'input': X}, {'target': X}, \
				n_epoch=num_epochs, validation_set=(X, X), \
				batch_size=batch_size, run_id='vae')
	    self.training_model.save(model_filename)
	else:
	    self.training_model.load(model_filename)

	self._construct_encoder_model()
	self._construct_decoder_model()

    def _construct_encoder_model(self):
	self.encoder_model = tflearn.DNN(self.z, session=self.training_model.session)

    def _construct_decoder_model(self):
	input_noise = tflearn.input_data(shape=[None, self.latent_dim], name='input_noise')
	for i, d in enumerate(reversed(self.hidden_dim)):
	    if i == 0:
		decoder = tflearn.fully_connected(input_noise, d, activation=self.decoder_fn, scope='decoder_layer_%d' % (i + 1), reuse=True)
		decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1), reuse=True)
		decoder = tflearn.dropout(decoder, self.dropout_rate)
	    else:
		decoder = tflearn.fully_connected(decoder, d, activation=self.decoder_fn, scope='decoder_layer_%d' % (i + 1), reuse=True)
		decoder = tflearn.batch_normalization(decoder, scope='decoder_bn_%d' % (i + 1), reuse=True)
		decoder = tflearn.dropout(decoder, self.dropout_rate)

	decoder = tflearn.fully_connected(decoder, self.input_dim, activation=self.squashing_fn, scope='decoder_output', reuse=True)
	self.decoder_model = tflearn.DNN(decoder, session=self.training_model.session)

    def encode(self, X):
	latent_z = self.encoder_model.predict({'input': X})
	return latent_z

    def decode(self, Z):
	x_reconstructed = self.decoder_model.predict({'input_noise': Z})
	return x_reconstructed

