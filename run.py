import argparse
import input_quickdraw

from vae import VAE

parser = argparse.ArgumentParser()

parser.add_argument('--input_dim', type=int, action='store', dest='input_dim')
parser.add_argument('--latent_dim', type=int, action='store', dest='latent_dim')
parser.add_argument('--hidden_dim', type=int, nargs='+', action='store', dest='hidden_dim')
parser.add_argument('--encoder_fn', default='sigmoid', action='store', dest='encoder_fn')
parser.add_argument('--decoder_fn', default='relu', action='store', dest='decoder_fn')
parser.add_argument('--squashing_fn', default='sigmoid', action='store', dest='squashing_fn')

parameters = parser.parse_args()

single = input_quickdraw.Single(['alarm_clock', 'broom', 'banana', 'lion', 'lollipop'], 5000)
X = single.train_data
Y = single.train_label

vae = VAE(parameters.input_dim, parameters.hidden_dim, parameters.latent_dim, encoder_fn=parameters.encoder_fn, decoder_fn=parameters.decoder_fn)
vae.train(X, batch_size=500, num_epochs=100, rerun=True)
