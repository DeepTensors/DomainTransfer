#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:23:43 2018

@author: anmol
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Domain Transfer Network Model
class Model(object):
    
    # Constructor
    # Content Extractor
    # Generator
    # Discriminator
    # Building final model
    
    # CONSTRUCTOR
    def __init__(self , mode , learning_rate):
        self.mode = mode
        self.learning_rate = learning_rate
        
    # Content Extractor
    def content_extractor(self , images , reuse = False):

        # Images: (batch_size , 32 , 32 , 3) && (batch_size , 32 , 32 , 1)
        # if image is in grayscale then covert it into rgb image
        if images.get_shape()[3] == 1:
            images = tf.image.grayscale_to_rgb(images)
            
        # Defining the scope of variables as content-extractor
        with tf.variable_scope('content_extractor' , reuse = reuse):
            # making the slim arg scope for default values
            with slim.arg_scope([slim.conv2d] , padding = 'SAME' , activation_fn = None,
                                stride = 2 , weight_initializer = tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm] , decay = 0.95 , center = True , scale = True , 
                                    activation_fn = tf.nn.relu , is_training = (self.mode == 'train' or self.mode == 'pretrain')):
                    
                    # Conv layer 1
                    # Output -> (batch_size , 16 , 16 , 64)
                    net = slim.conv2d(images , 64 , [3,3] , scope = 'conv1')
                    
                    # Batch Norm 1
                    net = slim.batch_norm(net , scope = 'bn1')
                    
                    # Conv Layer 2
                    # Output -> (batch_size , 8 , 8 , 128)
                    net = slim.conv2d(net , 128 , [3,3] , scope = 'conv2')
                    
                    # Batch Norm 2
                    net = slim.batch_norm(net , scope = 'bn2')
                    
                    # Conv Layer 3
                    # Output -> (batch_size , 4 , 4 , 256)
                    net = slim.conv2d(net , 256 , [3,3] , scope = 'conv3')
                    
                    # Batch Norm 3
                    net = slim.batch_norm(net , scope = 'bn3')
                    
                    # Conv Layer 4
                    # Output -> (batch_size , 1 , 1 , 128)
                    net = slim.conv2d(net , 128 , [4,4] , padding = 'VALID' , scope = 'conv4')
                    
                    # Batch Norm 4
                    # Activation -> tanh
                    net = slim.batch_norm(net , activation_fn = tf.nn.tanh , scope = 'bn4')
            
                    # self mode -> pretrain
                    if self.mode == 'pretrain':
                        
                        # Conv layer
                        # Output -> (batch_size , 1, 1 ,10)
                        net = slim.conv2d(net , 10 , [1,1] , padding = 'VALID' , scope = 'out')
                        
                        # Flattening Layer
                        net = slim.flatten(net)
            
                    return net

    # Generator
    # inputs -> (batch_size ,1,1,128)
    def generator(self , inputs , reuse = False):
        
        # Scope of variables -> generator
        with tf.variable_scope('generator' , reuse = reuse):
            # making the slim arg scope for default values
            with slim.arg_scope([slim.conv2d_transpose] , padding = 'SAME' , activation_fn = None,
                                stride = 2 , weights_initializer = tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm] , decay = 0.95 , center = True , scale = True , 
                                    activation_fn = tf.nn.relu , is_training = (self.mode == 'train')):
                    
                    # Conv Transpose1
                    # Output -> (batch_size , 4 , 4, 512)
                    net = slim.conv2d_transpose(inputs , 512 , [4,4] , padding = 'VALID' , scope = 'conv_transpose1')
                    
                    # Batch Norm
                    net = slim.batch_norm(net , scope = 'bn1')
                    
                    # Conv Transpose2
                    # Outpur -> (batch_size , 8 , 8 , 256)
                    net = slim.conv2d_transpose(net , 256 , [3,3] , scope = 'conv_transpose2')
                        
                    # Batch Norm
                    net = slim.batch_norm(net , scope = 'bn2')
                    
                    # Conv Transpose3
                    # Output -> (batch_size , 16 , 16 , 128)
                    net = slim.conv2d_transpose(net , 128 ,[3,3] , scope ='conv_transpose3')
                    
                    # Batch Norm
                    net = slim.batch_norm(net , scope = 'bn3')
                    
                    # Conv Transpose 4
                    # Activation function -> tanh
                    # Output ->( batch_Size , 32 , 32 , 1)
                    net = slim.conv2d_transpose(net , 1 , [3,3] , activation_fn = tf.nn.tanh , scope = 'conv_transpose4')
                    
                    # Return
                    return net