from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
import os
import numpy as np
import random
import cv2
import time


def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb



'''
A generator that yields a batch of (data, class_one, class_two).

输入:
        data_list  : 图片地址, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        
Output:
        (X_batch, Y1_batch, Y2_batch)
'''
def generator_batch_multitask(data_list, nbr_class_one=250, nbr_class_two=7, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):


    N = len(data_list)
    #打乱顺序
    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        #循环使用
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                label_one = int(line[-2])
                label_two = int(line[-1])
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1

        '''
        if preprocess:
            for i in range(current_batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image=True,
                                               result_size=(img_width, img_height))
        '''
        
        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if return_label:
            yield ([X_batch], [Y_batch_one, Y_batch_two])
        else:
            yield X_batch


def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of ([anchor, positive, negative], [class_one, class_two, pseudo_label]).

    Input:
        data_list  : a list of [img_path, vehicleID, modelID, colorID]
        dic_data_list: a dictionary: {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size
        mode       : generator used as for 'train', 'val'.
                     if mode is set 'train', dic_data_list has to be specified.
                     if mode is set 'val', dic_data_list could be a null dictionary: { }.
                     if mode is et 'feature_extraction', then return (X_anchor)


    Output:
        ([anchor, positive, negative], [class_one, class_two, pseudo_label]
    '''
    if shuffle:
        random.shuffle(data_list)

    N = len(data_list)
    dic = dic_data_list

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_anchor = np.zeros((current_batch_size, img_width, img_height, 3))
        X_positive = np.zeros((current_batch_size, img_width, img_height, 3))
        X_negative = np.zeros((current_batch_size, img_width, img_height, 3))

        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))
        Y_batch_fake = np.zeros((current_batch_size, 1))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print(line)
            anchor_path, vehicleID, modelID, colorID = line

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)

            anchor = scale_byRatio(anchor_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            if mode == 'train':
                # Find the same modelID, note that it is still a dictionary.
                # In this dictionary, the keys are colorIDs. In other words,
                # those images with the same modelID and the same colorID, the same
                # vehicleID are positives. Negatives are the ones with same modelID,
                # the same colorID but different vehicleIDs. Same modelID but different
                # colorIDs may also be considered as negatives.

                assert len(dic[modelID][colorID][vehicleID]) > 1, 'vehicleID: {} has only ONE image! The list is  {}'.format(vehicleID, dic[modelID][colorID][vehicleID])

                # copy a list of image paths with same vehicleID
                positive_list = dic[modelID][colorID][vehicleID][:]

                positive_list.remove(anchor_path)
                positive_path = random.choice(positive_list)
                positive = scale_byRatio(positive_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                negative_vehicleID_list = list(dic[modelID][colorID].keys())[:]
                negative_vehicleID_list.remove(vehicleID)
                assert negative_vehicleID_list !=[ ], 'vehicleID_list is [ ], {}'.format(dic[modelID][colorID].keys())
                negative_vehicleID = random.choice(negative_vehicleID_list)
                negative_path = random.choice(dic[modelID][colorID][negative_vehicleID])
                negative = scale_byRatio(negative_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = positive
                X_negative[i - current_index] = negative

            elif mode == 'val':
                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = anchor
                X_negative[i - current_index] = anchor

            if return_label:
                label_one = int(line[-2])
                label_two = int(line[-1])
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1



        X_anchor = X_anchor.astype(np.float64)
        X_positive = X_positive.astype(np.float64)
        X_negative = X_negative.astype(np.float64)
        X_anchor = preprocess_input(X_anchor)
        X_positive = preprocess_input(X_positive)
        X_negative = preprocess_input(X_negative)

        if save_network_input:
            print('X_anchor.shape: {}'.format(X_anchor.shape))
            X_anchor_to_save = X_anchor.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_anchor_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_anchor_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_anchor_to_save[:, :, 2], delimiter = ' ')

        if return_label:
            yield ([X_anchor, X_positive, X_negative], [Y_batch_one, Y_batch_two, Y_batch_fake])
        else:
            if mode == 'feature_extraction':
                yield X_anchor
            else:
                yield [X_anchor, X_positive, X_negative]



def filter_data_list(data_list):
    # data_list  : a list of [img_path, vehicleID, modelID, colorID]
    # {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    # dic helps us to sample positive and negative samples for each anchor.
    # https://arxiv.org/abs/1708.02386
    # The original paper says that "only the hardest triplets in which the three images have exactly
    # the same coarse-level attributes (e.g. color and model), can be used for similarity learning."
    dic = { }
    # We construct a new data list so that we could sample enough positives and negatives.
    new_data_list = [ ]
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        dic.setdefault(modelID, { })
        dic[modelID].setdefault(colorID, { })
        dic[modelID][colorID].setdefault(vehicleID, [ ]).append(imgPath)

    # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        #print(imgPath, vehicleID, modelID, colorID)
        if modelID in dic and colorID in dic[modelID] and vehicleID in dic[modelID][colorID] and \
                                                      len(dic[modelID][colorID][vehicleID]) == 1:
            dic[modelID][colorID].pop(vehicleID, None)

    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        if modelID in dic and colorID in dic[modelID] and len(dic[modelID][colorID].keys()) == 1:
            dic[modelID].pop(colorID, None)

    for modelID in dic:
        for colorID in dic[modelID]:
            for vehicleID in dic[modelID][colorID]:
                for imgPath in dic[modelID][colorID][vehicleID]:
                    new_data_list.append('{} {} {} {}'.format(imgPath, vehicleID, modelID, colorID))

    print('The original data list has {} samples, the new data list has {} samples.'.format(
                                 len(data_list), len(new_data_list)))
    return new_data_list, dic



