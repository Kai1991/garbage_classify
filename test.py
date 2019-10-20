from PIL import Image
import numpy as np


def test():
    '''
    import cv2
    img = cv2.imread('/home/yons/data/机器图像算法赛道-天气识别/Test_tmp/23110f60ff424b0dbdfa10b78aaad14f.webp')
    '''
    im = Image.open('/home/yons/data/机器图像算法赛道-天气识别/Test_tmp/23110f60ff424b0dbdfa10b78aaad14f.webp')
    if im.mode=="RGBA":
        im.load()  # required for png.split()
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
        im = background
    im.save('{}.jpg'.format("test"),'JPEG')
    

def test_1():
    from keras.utils import np_utils
    labels = [1]
    labels = np_utils.to_categorical(labels, 9)
    print(labels)
    print(np.argmax(labels[0], axis=0))


if __name__ == "__main__":
    test_1()