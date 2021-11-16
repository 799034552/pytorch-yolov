import os
import random
import xml.etree.ElementTree as ET
from const import CONST
import json
#=================================
# 将数据集处理成json格式并分开数据集、验证集。测试集
#==================================
xmlPath = CONST.xmlPath
train_test = 9
train_val = 9
def getImgData(filePath):
    root=ET.parse(xmlPath + "/" + filePath).getroot()
    picInfo = {
        "id": filePath[:filePath.index(".xml")],
        "boxes": [],
        "clsTypes": []
    }
    for obj in root.iter('object'):
        # 不使用difficult的数据
        if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
            continue
        cls = obj.find('name').text # 类名
        if cls not in CONST.classes:
            continue
        clsType = CONST.classes.index(cls)
        xmlbox = obj.find('bndbox')
        box = (int(xmlbox.find("xmin").text), int(xmlbox.find("ymin").text), int(xmlbox.find("xmax").text), int(xmlbox.find("ymax").text))
        picInfo["boxes"].append(box)
        picInfo["clsTypes"].append(clsType)
    return picInfo

if __name__ == "__main__":
    # 获取数据集数据
    random.seed(1)
    xmlFile = os.listdir(xmlPath)
    xmlFile = [x for x in xmlFile if x.find(".xml") != -1]
    imgDatas = []
    # 解析xml
    for item in xmlFile:
       imgDatas.append(getImgData(item))
    random.shuffle(imgDatas)
    test = imgDatas[:len(imgDatas) // train_test]
    imgDatas = imgDatas[len(imgDatas) // train_test:]
    val = imgDatas[: len(imgDatas) // train_val]
    train = imgDatas[len(imgDatas) // train_val:]
    res = {
        "train": train,
        "val": val,
        "test": test
    }
    # res = json.dumps(res, sort_keys=False,)
    with open("./data.json","w") as f:
        json.dump(res,f)
        print("加载入文件完成...")










