# -*- coding: utf-8 -*-
import os
import shutil
import codecs
import numpy as np
from glob import glob

from PIL import Image
import tensorflow as tf
from keras import backend
from keras.optimizers import adam, Nadam

from tensorflow.python.saved_model import tag_constants

from train import model_fn
from save_model import load_weights
import pandas as pd

backend.set_image_data_format('channels_last')


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


def preprocess_img(img_path, img_size):
    """
    image preprocessing
    you can add your special preprocess mothod here
    """

    img = Image.open(img_path)
    resize_scale = img_size / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)

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


    img = center_img(img, img_size)
    return img


def load_test_data(FLAGS):
    img_names = glob(FLAGS.test_data_local+'*')
    test_data = np.ndarray((len(img_names), FLAGS.input_size, FLAGS.input_size, 3),
                           dtype=np.uint8)
    for index, file_path in enumerate(img_names):
        tmp = file_path.split('.')[-1]
        if tmp != 'png' and tmp != 'jpg' and tmp != 'jpeg':
            print(file_path)  
        test_data[index] = preprocess_img(os.path.join(FLAGS.test_data_local, file_path), FLAGS.input_size)
    return img_names, test_data


def test_single_h5(FLAGS, h5_weights_path):
    if not os.path.isfile(h5_weights_path):
        print('%s is not a h5 weights file path' % h5_weights_path)
        return
    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    load_weights(model, FLAGS.eval_weights_path)
    img_names, test_data = load_test_data(FLAGS)
    predictions = model.predict(test_data, verbose=0)

    test_labels = []
    for index, pred in enumerate(predictions):
        pred_label = np.argmax(pred, axis=0)
        test_labels.append(pred_label + 1)
    img_names = [x.split('/')[-1] for x in img_names] 
    df = pd.DataFrame({"FileName":img_names,"type":test_labels})
    df.to_csv('result.csv',index=0)


def main():
    #tf.app.flags.DEFINE_string('eval_weights_path', '/home/yons/code/tmp_garbage/garbage_classify/model_snapshots/weights_022_0.7455.h5', 'weights file path need to be evaluate')
    tf.app.flags.DEFINE_string('eval_weights_path', '/home/yons/code/tmp_garbage/model_snapshots/weights_025_0.7397.h5', 'weights file path need to be evaluate')
    tf.app.flags.DEFINE_string('test_data_local', '/home/yons/data/机器图像算法赛道-天气识别/Test/', 'the test data path on obs')
    tf.app.flags.DEFINE_integer('input_size', 456, 'the input image size of the model') 
    tf.app.flags.DEFINE_float('learning_rate',1e-4, '')
    tf.app.flags.DEFINE_integer('num_classes', 9, 'the num of classes which your task should classify')


    FLAGS = tf.app.flags.FLAGS

    test_single_h5(FLAGS, FLAGS.eval_weights_path)


def test():
    '''
    import cv2
    img = cv2.imread('/home/yons/data/机器图像算法赛道-天气识别/Test_tmp/23110f60ff424b0dbdfa10b78aaad14f.webp')
    '''
    im = Image.open('/home/yons/data/机器图像算法赛道-天气识别/Test/23110f60ff424b0dbdfa10b78aaad14f.webp')
    if im.mode=="RGBA":
        im.load()  # required for png.split()
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
        im = background
    im.save('{}.jpg'.format("test"),'JPEG')
    


if __name__ == "__main__":
    main()