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
        self.config_file = 'config15/test_all_class.txt' 
        return
    
    def start(self):

        self._load_config(self.config_file)
        self._load_network()
        self._create_session()
        self._load_models()
        # self._terminate_session()


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
        label_shape = config_slice['label_shape']
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
            'predicty'      : predicty,
            'proby'         : proby,
            'net_name'      : net_name,
            'data_shape'    : data_shape,
            'label_shape'   : label_shape,
            'class_num'     : class_num,
            'net'           : net,
            'x'             : x
        }


    def _set_network(self, config_net, nn):

        print('Setting networks')

        if(config_net):
            net_type    = config_net['net_type']
            net_name    = config_net['net_name']
            data_shape  = config_net['data_shape']
            label_shape = config_net['label_shape']
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

            return {
                'predicty'      : predicty,
                'proby'         : proby,
                'net_name'      : net_name,
                'data_shape'    : data_shape,
                'label_shape'   : label_shape,
                'net'           : net,
                'class_num'     : class_num,
                'x'             : x
            }

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
            net_vars = [x for x in self.all_vars if x.name[0:len(n_config['net_name']) + 1]==n_config['net_name'] + '/']
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

    def start_test(self):

        test_slice_direction = self.config_test.get('test_slice_direction', 'all')
        save_folder = self.config_data['save_folder']
        test_time = []
        struct = ndimage.generate_binary_structure(3, 2)
        margin = self.config_test.get('roi_patch_margin', 5)

        for i in range(self.image_num):
            
            [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = self.dataloader.get_image_data_with_name(i)
            t0 = time.time()

            # test 1st network
            params1 = self.test_config(self.config_net1, self.net_config[0])
            pred1 = self._test_prob_and_pred(temp_imgs, temp_weight, params1, 3, self.net_config[0])

            wt_threshold = 2000
            if(self.config_test.get('whole_tumor_only', False) is True):
                pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
                pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
                out_label = pred1_lc
            else:
                # 5.2, test of 2nd network
                if(pred1.sum() == 0):
                    print('net1 output is null', temp_name)
                    bbox1 = get_ND_bounding_box(temp_imgs[0] > 0, margin)
                else:
                    pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
                    pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
                    bbox1 = get_ND_bounding_box(pred1_lc, margin)
                sub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox1[0], bbox1[1]) for one_img in temp_imgs]
                sub_weight = crop_ND_volume_with_bounding_box(temp_weight, bbox1[0], bbox1[1])

                params2 = self.test_config(self.config_net2, self.net_config[1])
                pred2 = self._test_prob_and_pred(sub_imgs, sub_weight, params2, 1, self.net_config[1])

                # 5.3, test of 3rd network
                if(pred2.sum() == 0):
                    [roid, roih, roiw] = sub_imgs[0].shape
                    bbox2 = [[0,0,0], [roid-1, roih-1, roiw-1]]
                    subsub_imgs = sub_imgs
                    subsub_weight = sub_weight
                else:
                    pred2_lc = ndimage.morphology.binary_closing(pred2, structure = struct)
                    pred2_lc = get_largest_two_component(pred2_lc)
                    bbox2 = get_ND_bounding_box(pred2_lc, margin)
                    subsub_imgs = [crop_ND_volume_with_bounding_box(one_img, bbox2[0], bbox2[1]) for one_img in sub_imgs]
                    subsub_weight = crop_ND_volume_with_bounding_box(sub_weight, bbox2[0], bbox2[1])

                params3 = self.test_config(self.config_net3, self.net_config[2])
                pred3 = self._test_prob_and_pred(subsub_imgs, subsub_weight, params3, 1, self.net_config[2])


                # 5.4, fuse results at 3 levels
                # convert subsub_label to full size (non-enhanced)
                label3_roi = np.zeros_like(pred2)
                label3_roi = set_ND_volume_roi_with_bounding_box_range(label3_roi, bbox2[0], bbox2[1], pred3)
                label3 = np.zeros_like(pred1)
                label3 = set_ND_volume_roi_with_bounding_box_range(label3, bbox1[0], bbox1[1], label3_roi)

                label2 = np.zeros_like(pred1)
                label2 = set_ND_volume_roi_with_bounding_box_range(label2, bbox1[0], bbox1[1], pred2)

                label1_mask = (pred1 + label2 + label3) > 0
                label1_mask = ndimage.morphology.binary_closing(label1_mask, structure = struct)
                label1_mask = get_largest_two_component(label1_mask, False, wt_threshold)
                label1 = pred1 * label1_mask
                
                label2_3_mask = (label2 + label3) > 0
                label2_3_mask = label2_3_mask * label1_mask
                label2_3_mask = ndimage.morphology.binary_closing(label2_3_mask, structure = struct)
                label2_3_mask = remove_external_core(label1, label2_3_mask)
                if(label2_3_mask.sum() > 0):
                    label2_3_mask = get_largest_two_component(label2_3_mask)
                label1 = (label1 + label2_3_mask) > 0
                label2 = label2_3_mask
                label3 = label2 * label3
                vox_3  = np.asarray(label3 > 0, np.float32).sum()
                if(0 < vox_3 and vox_3 < 30):
                    label3 = np.zeros_like(label2)

                # 5.5, convert label and save output
                out_label = label1 * 2
                if('Flair' in self.config_data['modality_postfix'] and 'mha' in self.config_data['file_postfix']):
                    out_label[label2>0] = 3
                    out_label[label3==1] = 1
                    out_label[label3==2] = 4
                elif('flair' in self.config_data['modality_postfix'] and 'nii' in self.config_data['file_postfix']):
                    out_label[label2>0] = 1
                    out_label[label3>0] = 4
                out_label = np.asarray(out_label, np.int16)

            test_time.append(time.time() - t0)
            final_label = np.zeros(temp_size, np.int16)
            final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
            save_array_as_nifty_volume(final_label, save_folder+"/{0:}.nii.gz".format(temp_name), img_names[0])
            print(temp_name)

        test_time = np.asarray(test_time)
        print('test time', test_time.mean())
        np.savetxt(save_folder + '/test_time.txt', test_time)
        
        return

    def test_config(self, config_net, n_config):

        if(config_net):
            data_shapes  = [ n_config['data_shape'][:-1],  n_config['data_shape'][:-1], n_config['data_shape'][:-1]]
            label_shapes = [ n_config['label_shape'][:-1],  n_config['label_shape'][:-1], n_config['label_shape'][:-1]]
            nets = [n_config['net'], n_config['net'], n_config['net']]
            outputs = [n_config['proby'], n_config['proby'], n_config['proby']]
            inputs =  [n_config['x'], n_config['x'], n_config['x']]
            class_num = n_config['class_num']
        else:
            data_shapes  = [ n_config['ax']['data_shape'][:-1],  n_config['sg']['data_shape'][:-1],  n_config['cr']['data_shape'][:-1]]
            label_shapes = [n_config['ax']['label_shape'][:-1], n_config['sg']['label_shape'][:-1], n_config['cr']['label_shape'][:-1]]
            nets = [n_config['ax']['net'], n_config['sg']['net'], n_config['cr']['net']]
            outputs = [n_config['ax']['proby'], n_config['sg']['proby'], n_config['cr']['proby']]
            inputs =  [n_config['ax']['x'], n_config['sg']['x'], n_config['cr']['x']]
            class_num = n_config['ax']['class_num']

        return {
            'data_shapes'   : data_shapes,
            'label_shapes'  : label_shapes,
            'nets'          : nets,
            'outputs'       : outputs,
            'inputs'        : inputs,
            'class_num'     : class_num
        }


    def _test_prob_and_pred(self, temp_imgs, temp_weight, params, shape_mode, n_config):

        prob = test_one_image_three_nets_adaptive_shape(temp_imgs, params['data_shapes'], 
                        params['label_shapes'], n_config['ax']['data_shape'][-1], params['class_num'],
                        self.batch_size, self.sess, params['nets'], params['outputs'], 
                        params['inputs'], shape_mode = shape_mode)

        pred =  np.asarray(np.argmax(prob, axis = 3), np.uint16)
        pred = pred * temp_weight

        return pred


if __name__ == '__main__':

    tumor_detector = TumorDetector()

    tumor_detector.start()

    tumor_detector.load_data()

    tumor_detector.start_test()
    


