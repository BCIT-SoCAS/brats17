# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import numpy as np
import nibabel
from PIL import Image
import random
from scipy import ndimage

def search_file_in_folder_list(folder_list, file_name):
    """
    Find the full filename from a list of folders
    inputs:
        folder_list: a list of folders
        file_name:  filename
    outputs:
        full_file_name: the full filename
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name

def load_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    return data

def save_array_as_nifty_volume(data, filename):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume
        
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode, bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'full': the entire roi should be inside the input volume
                     'valid': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center    

def transpose_volumes(volumes, slice_direction):
    """
    transpose a list of volumes
    inputs:
        volumes: a list of nd volumes
        slice_direction: 'axial', 'sagittal', or 'coronal'
    outputs:
        tr_volumes: a list of transposed volumes
    """
    if (slice_direction == 'axial'):
        tr_volumes = volumes
    elif(slice_direction == 'sagittal'):
        tr_volumes = [np.transpose(x, (2, 0, 1)) for x in volumes]
    elif(slice_direction == 'coronal'):
        tr_volumes = [np.transpose(x, (1, 0, 2)) for x in volumes]
    else:
        print('undefined slice direction:', slice_direction)
        tr_volumes = volumes
    return tr_volumes


def resize_ND_volume_to_given_shape(volume, out_shape, order = 3):
    """
    resize an nd volume to a given shape
    inputs:
        volume: the input nd volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized nd volume with given shape
    """
    shape0=volume.shape
    assert(len(shape0) == len(out_shape))
    scale = [(out_shape[i] + 0.0)/shape0[i] for i in range(len(shape0))]
    out_volume = ndimage.interpolation.zoom(volume, scale, order = order)
    return out_volume

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

class DataLoader():
    def __init__(self):
        pass
        
    def set_params(self, config):
        self.config = config
        self.data_root = config['data_root']
        self.modality_postfix = config['modality_postfix']
        self.intensity_normalize = config.get('intensity_normalize', None)
        self.label_postfix =  config.get('label_postfix', None)
        self.file_postfix = config['file_post_fix']
        self.data_names = config['data_names']
        self.data_num = config.get('data_num', 0)
        self.data_resize = config.get('data_resize', None)
        self.with_ground_truth  = config.get('with_ground_truth', False)
        self.with_flip = config.get('with_flip', False)
        self.label_convert_source = self.config.get('label_convert_source', None)
        self.label_convert_target = self.config.get('label_convert_target', None)
        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
        if(self.intensity_normalize == None):
            self.intensity_normalize = [True] * len(self.modality_postfix)
            
    def __get_patient_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient 
        names from that file, otherwise get search all the names automatically in data_root
        """
        if(self.data_names):
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content] 
        else: # load all image in data_root
            sub_dirs = [x[0] for x in os.walk(self.data_root[0])]
            patient_names = []
            for sub_dir in sub_dirs:
                names = os.listdir(sub_dir)
                if(sub_dir == self.data_root[0]):
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = x[:idx]
                            sub_patient_names.append(xsplit)
                else:
                    sub_dir_name = sub_dir[len(self.data_root[0])+1:]
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = os.path.join(sub_dir_name,x[:idx])
                            sub_patient_names.append(xsplit)                    
                sub_patient_names = list(set(sub_patient_names))
                sub_patient_names.sort()
                patient_names.extend(sub_patient_names)   
        return patient_names
    
    def load_data(self):
        """
        load all the training/testing data
        """
        self.patient_names = self.__get_patient_names()
        assert(len(self.patient_names)  > 0)
        X = []
        W = []
        Y = []
        P = []
        data_num = self.data_num if (self.data_num) else len(self.patient_names)
        for i in range(data_num):
            volume_list = []
            for mod_idx in range(len(self.modality_postfix)):
                volume_name_short = self.patient_names[i] + '_' + self.modality_postfix[mod_idx] + '.' + self.file_postfix
                volume_name = search_file_in_folder_list(self.data_root, volume_name_short)
                volume = load_nifty_volume_as_array(volume_name)
                if(self.data_resize):
                    volume = resize_3D_volume_to_given_shape(volume, self.data_resize, 1)
                if(mod_idx == 0):
                    weight = np.asarray(volume > 0, np.float32)
                if(self.intensity_normalize[mod_idx]):
                    volume = itensity_normalize_one_volume(volume)
                volume_list.append(volume)
            X.append(volume_list)
            W.append(weight)
            if(self.with_ground_truth):
                label_name_short = self.patient_names[i] + '_' + self.label_postfix + '.' + self.file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
                label = load_nifty_volume_as_array(label_name)
                if(self.data_resize):
                    label = resize_3D_volume_to_given_shape(label, self.data_resize, 0)
                Y.append(label)
            if(i%50 == 0 or i == data_num):
                print('{0:}/{1:} volumes have been loaded'.format(i, data_num))
        self.data   = X
        self.weight = W
        self.label  = Y
    
    def get_subimage_batch(self):
        """
        sample a batch of image patches for segmentation. Only used for training
        """
        flag = False
        while(flag == False):
            batch = self.__get_one_batch()
            labels = batch['labels']
            if(labels.sum() > 0):
                flag = True
        return batch
    
    def __get_one_batch(self):
        """
        get a batch from training data
        """
        batch_size = self.config['batch_size']
        data_shape = self.config['data_shape']
        label_shape = self.config['label_shape']
        down_sample_rate = self.config.get('down_sample_rate', 1.0)
        data_slice_number = data_shape[0]
        label_slice_number = label_shape[0]
        batch_sample_model   = self.config.get('batch_sample_model', ('full', 'valid', 'valid'))
        batch_slice_direction= self.config.get('batch_slice_direction', 'axial') # axial, sagittal, coronal or random
        train_with_roi_patch = self.config.get('train_with_roi_patch', False)
        keep_roi_outside = self.config.get('keep_roi_outside', False)
        if(train_with_roi_patch):
            label_roi_mask = self.config['label_roi_mask']
            roi_patch_margin  = self.config['roi_patch_margin']

        # return batch size: [batch_size, slice_num, slice_h, slice_w, moda_chnl]
        data_batch = []
        weight_batch = []
        prob_batch = []
        label_batch = []
        slice_direction = batch_slice_direction
        if(slice_direction == 'random'):
            directions = ['axial', 'sagittal', 'coronal']
            idx = random.randint(0,2)
            slice_direction = directions[idx]
        for i in range(batch_size):
            if(self.with_flip):
                flip = random.random() > 0.5
            else:
                flip = False
            self.patient_id = random.randint(0, len(self.data)-1)
            data_volumes = [x for x in self.data[self.patient_id]]
            weight_volumes = [self.weight[self.patient_id]]
            boundingbox = None
            if(self.with_ground_truth):
                label_volumes = [self.label[self.patient_id]]
                if(train_with_roi_patch):
                    mask_volume = np.zeros_like(label_volumes[0])
                    for mask_label in label_roi_mask:
                        mask_volume = mask_volume + (label_volumes[0] == mask_label)
                    [d_idxes, h_idxes, w_idxes] = np.nonzero(mask_volume)
                    [D, H, W] = label_volumes[0].shape
                    mind = max(d_idxes.min() - roi_patch_margin, 0)
                    maxd = min(d_idxes.max() + roi_patch_margin, D)
                    minh = max(h_idxes.min() - roi_patch_margin, 0)
                    maxh = min(h_idxes.max() + roi_patch_margin, H)
                    minw = max(w_idxes.min() - roi_patch_margin, 0)
                    maxw = min(w_idxes.max() + roi_patch_margin, W)
                    if(keep_roi_outside):
                        boundingbox = [mind, maxd, minh, maxh, minw, maxw]
                    else:
                        for idx in range(len(data_volumes)):
                            data_volumes[idx] = data_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]
                        for idx in range(len(weight_volumes)):
                            weight_volumes[idx] = weight_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]
                        for idx in range(len(label_volumes)):
                            label_volumes[idx] = label_volumes[idx][np.ix_(range(mind, maxd), 
                                                                     range(minh, maxh), 
                                                                     range(minw, maxw))]

                if(self.label_convert_source and self.label_convert_target):
                    label_volumes[0] = convert_label(label_volumes[0], self.label_convert_source, self.label_convert_target)
        
            transposed_volumes = transpose_volumes(data_volumes, slice_direction)
            volume_shape = transposed_volumes[0].shape
            sub_data_shape = [data_slice_number, data_shape[1], data_shape[2]]
            sub_label_shape =[label_slice_number, label_shape[1], label_shape[2]]
            center_point = get_random_roi_sampling_center(volume_shape, sub_label_shape, batch_sample_model, boundingbox)
            sub_data = []
            for moda in range(len(transposed_volumes)):
                sub_data_moda = extract_roi_from_volume(transposed_volumes[moda],center_point,sub_data_shape)
                if(flip):
                    sub_data_moda = np.flip(sub_data_moda, -1)
                if(down_sample_rate != 1.0):
                    sub_data_moda = ndimage.interpolation.zoom(sub_data_moda, 1.0/down_sample_rate, order = 1)   
                sub_data.append(sub_data_moda)
            sub_data = np.asarray(sub_data)
            data_batch.append(sub_data)
            transposed_weight = transpose_volumes(weight_volumes, slice_direction)
            sub_weight = extract_roi_from_volume(transposed_weight[0],
                                                  center_point,
                                                  sub_label_shape,
                                                  fill = 'zero')
            
            if(flip):
                sub_weight = np.flip(sub_weight, -1)
            if(down_sample_rate != 1.0):
                    sub_weight = ndimage.interpolation.zoom(sub_weight, 1.0/down_sample_rate, order = 1)   
            weight_batch.append([sub_weight])
            if(self.with_ground_truth):
                tranposed_label = transpose_volumes(label_volumes, slice_direction)
                sub_label = extract_roi_from_volume(tranposed_label[0],
                                                     center_point,
                                                     sub_label_shape,
                                                     fill = 'zero')
                if(flip):
                    sub_label = np.flip(sub_label, -1)
                if(down_sample_rate != 1.0):
                    sub_label = ndimage.interpolation.zoom(sub_label, 1.0/down_sample_rate, order = 0)  
                label_batch.append([sub_label])
                    
        data_batch = np.asarray(data_batch, np.float32)
        weight_batch = np.asarray(weight_batch, np.float32)
        label_batch = np.asarray(label_batch, np.int64)
        prob_batch = np.asarray(prob_batch, np.float32)
        batch = {}
        batch['images']  = np.transpose(data_batch, [0, 2, 3, 4, 1])
        batch['weights'] = np.transpose(weight_batch, [0, 2, 3, 4, 1])
        batch['labels']  = np.transpose(label_batch, [0, 2, 3, 4, 1])
        
        return batch
    
    # The following two functions are used for testing
    def get_total_image_number(self):
        return len(self.data)
    
    def get_image_data_with_name(self, i):
        return [self.data[i], self.weight[i], self.patient_names[i]]