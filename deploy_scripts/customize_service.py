import ast
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.saved_model import tag_constants
from model_service.tfserving_model_service import TfServingBaseService


class garbage_classify_service(TfServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 456  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.get_default_graph().as_default():
            self.sess = tf.Session(graph=tf.Graph(), config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_path)
            self.signature = meta_graph_def.signature_def

            # define input and out tensor of your model here
            input_images_tensor_name = self.signature[self.signature_key].inputs[self.input_key_1].name
            output_score_tensor_name = self.signature[self.signature_key].outputs[self.output_key_1].name
            self.input_images = self.sess.graph.get_tensor_by_name(input_images_tensor_name)
            self.output_score = self.sess.graph.get_tensor_by_name(output_score_tensor_name)

        self.label_id_name_dict = \
            {
                "0":"雨凇",
                "1":"雾凇",
                "2":"雾霾",
                "3":"霜",
                "4":"露",
                "5":"结冰",
                "6":"降雨",
                "7":"降雪",
                "8":"冰雹"
            }

    def center_img(self, img, size=None, fill_value=255):
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

    def preprocess_img(self, img):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        resize_scale = self.input_size / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)
        img = self.center_img(img, self.input_size)
        return img
        #################################################################
        # img = img.resize((self.input_size, self.input_size))
        # img = img.convert('RGB')
        # img = np.array(img)
        # img = img.astype(np.float)
        # img = img[:, :, ::-1]
        # return img

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.preprocess_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data[self.input_key_1]
        img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
        img = np.asarray(img, np.float32) / 255.0
        mean = [0.50064302,0.52358398,0.50864987]
        std = [0.21226286,0.20765279,0.21247285]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]
        pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
        if pred_score is not None:
            pred_label = np.argmax(pred_score[0], axis=1)[0]
            result = {'result': self.label_id_name_dict[str(pred_label)]}
        else:
            result = {'result': 'predict score is None'}
        return result

    def _postprocess(self, data):
        return data
