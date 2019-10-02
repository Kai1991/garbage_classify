import os
dress="//home/yons/data/机器图像算法赛道-天气识别/clearTrain/Train/"
with open("train.txt","w") as f:
    for root,dirs,files in os.walk(dress):
        # root = root.replace(dress,'')
        for file in files:
            f.write(os.path.join(root, file) + "\n")
