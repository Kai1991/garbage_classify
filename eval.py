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

from train import model_fn, model_fn_SE_ResNet50
from save_model import load_weights

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
    label_files = glob(os.path.join(FLAGS.test_data_local, '*.txt'))
    test_data = np.ndarray((len(label_files), FLAGS.input_size, FLAGS.input_size, 3),
                           dtype=np.float32)
    img_names = []
    test_labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_names.append(line_split[0])
        test_data[index] = preprocess_img(os.path.join(FLAGS.test_data_local, line_split[0]), FLAGS.input_size)
        test_labels.append(int(line_split[1]))
    return img_names, test_data, test_labels


def test_single_h5(FLAGS, h5_weights_path):
    if not os.path.isfile(h5_weights_path):
        print('%s is not a h5 weights file path' % h5_weights_path)
        return
    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    #model = model_fn_SE_ResNet50(FLAGS, objective, optimizer, metrics)
    load_weights(model, FLAGS.eval_weights_path)
    img_names, test_data, test_labels = load_test_data(FLAGS)
    predictions = model.predict(test_data, verbose=0)

    right_count = 0
    error_infos = []

    print("img len :")
    print(len(img_names))
    print("test_data len :")
    print(len(test_data))
    print("test_labels len :")
    print(len(test_labels))

    for index, pred in enumerate(predictions):
        pred_label = np.argmax(pred, axis=0) + 1
        test_label = test_labels[index]
        if pred_label == test_label:
            print("{},{},{}".format(img_names[index],test_label,pred_label))
            right_count += 1
        else:
            error_infos.append('%s, %s, %s\n' % (img_names[index], test_label, pred_label))

    accuracy = right_count / len(img_names)
    print('accuracy: %s' % accuracy)
    result_file_name = os.path.join(os.path.dirname(h5_weights_path),
                                    '%s_accuracy.txt' % os.path.basename(h5_weights_path))
    with open(result_file_name, 'w') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, true_label, pred_label\n')
        f.writelines(error_infos)
        f.write('####################################\n')
        f.write('accuracy: %s\n' % accuracy)
    print('end')


def test_batch_h5(FLAGS):
    """
    test all the h5 weights files in the model_dir
    """
    file_paths = glob(os.path.join(FLAGS.eval_weights_path, '*.h5'))
    for file_path in file_paths:
        test_single_h5(FLAGS, file_path)


def test_single_model(FLAGS):
    if FLAGS.eval_pb_path.startswith('s3//'):
        pb_model_dir = '/cache/tmp/model'
        if os.path.exists(pb_model_dir):
            shutil.rmtree(pb_model_dir)
        shutil.copytree(FLAGS.eval_pb_path, pb_model_dir)
    else:
        pb_model_dir = FLAGS.eval_pb_path
    signature_key = 'predict_images'
    input_key_1 = 'input_img'
    output_key_1 = 'output_score'
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.get_default_graph().as_default():
        sess1 = tf.Session(graph=tf.Graph(), config=config)
        pb_model_dir1 = pb_model_dir 
        meta_graph_def = tf.saved_model.loader.load(sess1, [tag_constants.SERVING], pb_model_dir)
        if FLAGS.eval_pb_path.startswith('s3//'):
            shutil.rmtree(pb_model_dir)
        signature = meta_graph_def.signature_def
        input_images_tensor_name = signature[signature_key].inputs[input_key_1].name
        output_score_tensor_name = signature[signature_key].outputs[output_key_1].name

        input_images = sess1.graph.get_tensor_by_name(input_images_tensor_name)
        output_score = sess1.graph.get_tensor_by_name(output_score_tensor_name)


    img_names, test_data, test_labels = load_test_data(FLAGS)
    right_count = 0
    error_infos = []
    for index, img in enumerate(test_data):
        img = img[np.newaxis, :, :, :]
        pred_score = sess1.run([output_score], feed_dict={input_images: img})
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0], axis=1)[0]
            test_label = test_labels[index]
            if pred_label == test_label:
                right_count += 1
            else:
                error_infos.append('%s, %s, %s\n' % (img_names[index], test_label, pred_label))
        else:
            print('pred_score is None')
    accuracy = right_count / len(img_names)
    print('accuracy: %s' % accuracy)
    result_file_name = os.path.join(FLAGS.eval_pb_path, 'accuracy1.txt')
    with open(result_file_name, 'w') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, true_label, pred_label\n')
        f.writelines(error_infos)
        f.write('####################################\n')
        f.write('accuracy: %s\n' % accuracy)

    


def eval_model(FLAGS):
    if FLAGS.eval_weights_path != '':
        if os.path.isdir(FLAGS.eval_weights_path):
            test_batch_h5(FLAGS)
        else:
            test_single_h5(FLAGS, FLAGS.eval_weights_path)
    elif FLAGS.eval_pb_path != '':
        test_single_model(FLAGS)
