# This is a refactored code made by Jed Iquin from the following:
#
# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.

from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
from util.data_loader import *
from util.data_process import *
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory

class TumorDetector:

    def __init__(self):
        
        self.net_config= []
        self.net_vars = []
        self.nn = 3
        return
    
    def start(self):

        config_file = 'config15/test_all_class.txt'

        self._load_config(config_file)
        self._load_network()
        self._create_session()
        self._load_models()
        self._terminate_session()


    ####################################
    # LOADING CONFIGS AND MODELS
    ####################################
    def _load_config(self, config_file):

        print(f'Loading config file: {config_file}')
        
        # load the config file
        self.config = parse_config(config_file)
        self.config_data = self.config['data']
        self.config_net1 = self.config.get('network1', None)
        self.config_net3 = self.config.get('network3', None)
        self.config_net2 = self.config.get('network2', None)
        self.config_test = self.config['testing']  
        self.batch_size  = self.config_test.get('batch_size', 5)

        return

    def _load_network(self):

        print('Loading network')

        self.net_config.append(self._set_network(self.config_net1, 1))
        if(self.config_test.get('whole_tumor_only', False) is False):
            self.net_config.append(self._set_network(self.config_net2, 2))
            self.net_config.append(self._set_network(self.config_net3, 3))

    def _construct_graph(self, config_slice):

        print('Constructing graph')
        net_type    = config_slice['net_type']
        net_name    = config_slice['net_name']
        data_shape  = config_slice['data_shape']
        #label_shape = config_slice['label_shape']
        class_num   = config_slice['class_num']

        full_data_shape = [self.batch_size] + data_shape
        x = tf.placeholder(tf.float32, shape = full_data_shape)          
        net_class = NetFactory.create(net_type)
        net = net_class(num_classes = class_num,w_regularizer = None,
                    b_regularizer = None, name = net_name)
        net.set_params(config_slice)
        predicty = net(x, is_training = True)
        proby = tf.nn.softmax(predicty)

        return {
            'predicty'  : predicty,
            'proby'     : proby,
            'net_name'  : net_name
        }


    def _set_network(self, config_net, nn):

        print('Setting networks')

        if(config_net):
            net_type    = config_net['net_type']
            net_name    = config_net['net_name']
            data_shape  = config_net['data_shape']
            #label_shape = config_net['label_shape']
            class_num   = config_net['class_num']
            
            # construct graph for 1st network
            full_data_shape = [self.batch_size] + data_shape
            x = tf.placeholder(tf.float32, shape = full_data_shape)          
            net_class = NetFactory.create(net_type)
            net = net_class(num_classes = class_num,w_regularizer = None,
                        b_regularizer = None, name = net_name)
            net.set_params(config_net)
            predicty = net(x, is_training = True)
            proby = tf.nn.softmax(predicty)

            return predicty, proby, net_name

        else:

            config_netax = self.config[f'network{nn}ax']
            config_netsg = self.config[f'network{nn}sg']
            config_netcr = self.config[f'network{nn}cr']
               
            result = {
                'ax': self._construct_graph(config_netax),
                'sg': self._construct_graph(config_netsg),
                'cr': self._construct_graph(config_netcr),
            }

            result['ax']['config'] = config_netax
            result['sg']['config'] = config_netsg
            result['cr']['config'] = config_netcr
            
            return result 
            

    def _create_session(self):
        self.all_vars = tf.global_variables()
        self.sess = tf.InteractiveSession()   
        self.sess.run(tf.global_variables_initializer()) 
        return

    def _terminate_session(self):
        self.sess.close()
        return

    def _load_models(self):

        self._load_model(self.config_net1, self.net_config[0])
        if(self.config_test.get('whole_tumor_only', False) is False):
            self._load_model(self.config_net2, self.net_config[1])
            self._load_model(self.config_net3, self.net_config[2])

        return

    def _load_model(self, config_net, n_config):

        print('Loading model')
        if(config_net):
            net_vars = [x for x in self.all_vars if x.name[0:len(n_config[2]) + 1]==n_config[2] + '/']
            saver = tf.train.Saver(net_vars)
            saver.restore(self.sess, config_net['model_file'])
        else:
            netax_vars = [x for x in self.all_vars 
                if x.name[0:len(n_config['ax']['net_name']) + 1]==n_config['ax']['net_name'] + '/']
            saverax = tf.train.Saver(netax_vars)
            saverax.restore(self.sess, n_config['ax']['config']['model_file'])
            netsg_vars = [x for x in self.all_vars 
                if x.name[0:len(n_config['sg']['net_name']) + 1]==n_config['sg']['net_name'] + '/']
            saversg = tf.train.Saver(netsg_vars)
            saversg.restore(self.sess, n_config['sg']['config']['model_file'])     
            netcr_vars = [x for x in self.all_vars if 
                x.name[0:len(n_config['cr']['net_name']) + 1]==n_config['cr']['net_name'] + '/']
            savercr = tf.train.Saver(netcr_vars)
            savercr.restore(self.sess, n_config['cr']['config']['model_file'])

        return

    ####################################
    # LOADING DATA
    ####################################
    def load_data(self):
        """
        Loads the data specified in the loaded config file
        """

        self.dataloader = DataLoader(self.config_data)
        self.dataloader.load_data()
        self.image_num = self.dataloader.get_total_image_number()

        return

    ####################################
    # TESTING METHOD
    ####################################

    def start_test():
        

if __name__ == '__main__':

    tumor_detector = TumorDetector()

    tumor_detector.start()


