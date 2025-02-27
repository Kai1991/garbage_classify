# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from aug import augumentor
from utils import scale_byRatio

class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size, train=False):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=0):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def img_aug(self, img):
        data_gen = ImageDataGenerator()
        dic_parameter = {'flip_horizontal': random.choice([True, False]),
                         'flip_vertical': random.choice([True, False]),
                         'theta': random.choice([0, 0, 0, 90, 180, 270])
                        }


        img_aug = data_gen.apply_transform(img, transform_parameters=dic_parameter)
        return img_aug

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = scale_byRatio(img_path,return_width=256)

        # 数据增强
        if self.train:
            # img = self.img_aug(img)
            img = augumentor(img)
            #pass
        # 数据归一化
        img = np.asarray(img, np.float32) / 255.0
        mean = [0.50064302,0.52358398,0.50864987]
        std = [0.21226286,0.20765279,0.21247285]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]

        
        #img = self.center_img(img, self.img_size[0])
        return img


        ########################################
        # img = Image.open(img_path)
        # img = img.resize((self.img_size[0], self.img_size[0]))
        # img = img.convert('RGB')
        # img = np.array(img)
        # img = img.astype(np.float)
        # # if self.train:
        # #     # img = self.img_aug(img)
        # #     img = augumentor(img)
        # img = img[:, :, ::-1]
        #
        # return img
        ########################################
        # Img = Image.open(img_path)
        # Img = cv2.cvtColor(np.asarray(Img), cv2.COLOR_RGB2BGR)
        # Img = cv2.resize(Img, (self.img_size[0], self.img_size[0]))
        # Img = Img[:, :, (2, 1, 0)]
        # Img = np.asarray(Img)
        # Img = Img.astype(np.float)
        # return Img




    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)

# 标签平滑
def smooth_labels(y, smooth_factor=0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y

def data_flow(train_data_dir, batch_size, num_classes, input_size):  # need modify
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)

    labels = np_utils.to_categorical(labels, num_classes)
    # 标签平滑
    labels = smooth_labels(labels)

    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size=0.1, random_state=0)
    print('total samples: %d, training samples: %d, validation samples: %d' % (
        len(img_paths), len(train_img_paths), len(validation_img_paths)))

    print('total samples: %d, training samples: %d, validation samples: %d' % (len(img_paths), len(train_img_paths), len(validation_img_paths)))

    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size], True)
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size], False)

    return train_sequence, validation_sequence


if __name__ == '__main__':
    # train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator = data_flow(dog_cat_data_path, batch_size)
    # for i in range(10):
    #     train_data_batch = next(train_data_generator)
    # train_enqueuer.stop()
    # validation_enqueuer.stop()
    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size)
    batch_data, bacth_label = train_sequence.__getitem__(5)
    label_name = ['cat', 'dog']
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_2_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_3_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    print('end')
