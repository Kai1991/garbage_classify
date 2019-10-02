import os 
import pandas as pd 




def main(path='/home/yons/data/机器图像算法赛道-天气识别/clearTrain/Train_Labels'):
    names = []
    # 读取文件名
    for _,filename in enumerate(os.listdir(path)):
        names.append(os.path.splitext(filename)[0])


    df = pd.read_csv("/home/yons/data/机器图像算法赛道-天气识别/Train_label.csv")
    for _,row in df.iterrows():
        name = os.path.splitext(row['FileName'])[0]
        classtmp = int(row["type"]) - 1
        print(name)
   
        if name not in names:
            continue 

        with open(path + "/" + name +".txt","w+") as f:
            f.write("" + name + ".jpg" + ", " + str(classtmp))


if __name__ == "__main__":
    main()