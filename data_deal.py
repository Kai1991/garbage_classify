import os 
import pandas as pd
from skimage.transform import resize 
from skimage import io
from glob import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(path='/home/yons/data/机器图像算法赛道-天气识别/resize'):
    names = []
    # 读取文件名
    for _,filename in enumerate(os.listdir(path)):
        names.append(os.path.splitext(filename)[0])


    df = pd.read_csv("/home/yons/data/机器图像算法赛道-天气识别/Train_label.csv")
    for _,row in df.iterrows():
        name_tail = row['FileName']
        name = os.path.splitext(name_tail)[0]
        classtmp = int(row["type"])
        print(name)
   
        if name not in names:
            continue 

        with open(path + "/" + name +".txt","w+") as f:
            f.write("" + name+'.jpg' + ", " + str(classtmp))






MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image
def center_crop_and_resizes(path='/home/yons/data/机器图像算法赛道-天气识别/Train_nopng',
    image_size=512,to_path='/home/yons/data/机器图像算法赛道-天气识别/resize/'):
    #读取所有图片名字
    label_files = glob(os.path.join(path, '*.jpg'))
    for index, file_path in enumerate(label_files):
        try:
            name = file_path.split('/')[-1]
            img = io.imread(file_path)
            img = center_crop_and_resize(img,image_size)
            io.imsave(to_path  + name,img)
        except Exception:
            print(name)
        

if __name__ == "__main__":
    center_crop_and_resizes()
    #main()