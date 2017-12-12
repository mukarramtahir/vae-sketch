# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:09:48 2017

@author: atan10
"""
#import matplotlib
#matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time as time

# Import MNIST data
import input_quickdraw  

# Training Parameters
learning_rate = 0.01
k = 1

num_steps = 2500
batch_size = 200

display_step = 100

# Network Parameters
num_hidden_1 = 500 # 1st layer num features
num_hidden_2 = 2 # 2nd layer num features (latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
z_temp = tf.placeholder("float",[None,num_hidden_2])

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
        
        
# Building the encoder
def encoder(x,weights,biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2mn = tf.add(tf.matmul(layer_1, weights['encoder_h2_mn']),
                                   biases['encoder_b2_mn'])
    layer_2std = tf.add(tf.matmul(layer_1, weights['encoder_h2_std']),
                                   biases['encoder_b2_std'])
    
    #Generate a random distribution in latent layer
    noise = tf.random_normal([1,num_hidden_2])
    z = layer_2mn + tf.multiply(noise,tf.exp(0.5*layer_2std))
    model = {'z':z, 'layer_2mn': layer_2mn, 'layer_2std': layer_2std }
    return model


# Building the decoder
def decoder(x,weights,biases):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


weights = {
    'encoder_h1': tf.Variable(xavier_init(num_input,num_hidden_1)),
    'encoder_h2_mn': tf.Variable(xavier_init(num_hidden_1,num_hidden_2)),
    'encoder_h2_std': tf.Variable(xavier_init(num_hidden_1,num_hidden_2)),
    'decoder_h1': tf.Variable(xavier_init(num_hidden_2,num_hidden_1)),
    'decoder_h2': tf.Variable(xavier_init(num_hidden_1,num_input)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2_mn': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b2_std': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Construct model
encoder_op = encoder(X,weights,biases)
decoder_op = decoder(encoder_op['z'],weights,biases)
decoder_gen = decoder(z_temp,weights,biases)


# Prediction
y_pred = decoder_op
y_true = X


# Define loss and optimizer, minimize the squared error
log_likelihood = tf.reduce_sum(y_true*tf.log(y_pred + 1e-9)+(1 - y_true)*tf.log(1 - y_pred + 1e-9), reduction_indices=1)
KL_term = -0.5*tf.reduce_sum(1 + 2*encoder_op['layer_2std'] - tf.pow(encoder_op['layer_2mn'],2)
        - tf.exp(2*encoder_op['layer_2std']), reduction_indices=1)
log_likelihood = tf.reduce_mean(log_likelihood)
KL_term = tf.reduce_mean(k*KL_term)
variational_lower_bound = log_likelihood-KL_term
#optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)
#loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#log_likelihood = tf.reduce_mean(log_likelihood)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(-variational_lower_bound)
   


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    t1 = time.time()
    # Training
    for i in range(1, num_steps+1):
        batch_x, _ = input_quickdraw.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_x})
        if i % display_step == 0 or i == 1:
            #vlb_eval = variational_lower_bound.eval(feed_dict={X: batch_x})
            likelihood_eval = log_likelihood.eval(feed_dict={X: batch_x})
            KL_eval = KL_term.eval(feed_dict={X:batch_x})
            print(i,'\t',likelihood_eval,'\t',KL_eval)
    t2 = time.time()
    training_time = t2-t1
    print("training time: ", training_time)

    batch_x, batch_y = input_quickdraw.next_batch(input_quickdraw.n_samples)
    z_sub = {'bat': 1, 'bac': 1, 'bbt': 1, 'bbc': 1, 'but': 1, 'buc': 1, 'fat': 2, 'fac': 2, 'fbt': 2, 'fbc': 2, 'fut': 2, 'fuc': 2}
    z_obj = {'bat': 1, 'bac': 2, 'bbt': 1, 'bbc': 2, 'but': 1, 'buc': 2, 'fat': 1, 'fac': 2, 'fbt': 1, 'fbc': 2, 'fut': 1, 'fuc': 2}
    z_loc = {'bat': 3, 'bac': 3, 'bbt': 2, 'bbc': 2, 'but': 1, 'buc': 1, 'fat': 3, 'fac': 3, 'fbt': 2, 'fbc': 2, 'fut': 1, 'fuc': 1}
    # Encode and decode the digit image
    g = sess.run(encoder_op, feed_dict={X: batch_x})

    g = g['z']

    plt.clf()
    '''
    plt.subplot(1, 3, 1)
    yz = np.array([z_sub[ys] for ys in batch_y])
    plt.scatter(g[:, 0], g[:, 1], c=yz, s=1.5)
    plt.subplot(1, 3, 2)
    yz = np.array([z_obj[ys] for ys in batch_y])
    plt.scatter(g[:, 0], g[:, 1], c=yz, s=1.5)
    '''
    yz = np.array([z_loc[ys] for ys in batch_y])
    plt.scatter(g[:, 0], g[:, 1], c=yz, s=1.5)
    #plt.show()
    plt.savefig('latent_space.png')

    
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    plt.figure()        
    n = 5
    for i in range(n):
        # MNIST test set
        batch_x, _ = input_quickdraw.next_batch(1)
        generate_random = sess.run(tf.random_normal([1,num_hidden_2],stddev = 2))
	print(i, generate_random)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        rand = sess.run(decoder_gen, feed_dict = {z_temp:generate_random})

        x_image = np.reshape(batch_x,(28,28))
        plt.subplot(5,3,3*i+1)
        plt.imshow(x_image)
        
        x_reconstruction_image = (np.reshape(g, (28,28)))
        #plot it!
        plt.subplot(5,3,3*i+2)
        plt.imshow(x_reconstruction_image)
        
        x_random_image = (np.reshape(rand,(28,28)))
        plt.subplot(5,3,3*i+3)
        plt.imshow(x_random_image)
    #plt.show()
    plt.savefig('Quickdraw_test_images')
    
    
        
        
