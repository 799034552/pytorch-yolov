import json
import numpy as np
import random
import xml.etree.ElementTree as ET
from const import CONST
import json
import os
#=========================================================
#           处理原始coco数据集，生成txt文件
#=========================================================
def handleCOCO():
    # 处理验证集
    res = {}
    with open(f'{CONST.cocoPath}/instances_val2014.json', 'r') as f:
        data = json.load(f)
        classes = []
        # 读取类型
        for item in data["categories"]:
            classes.append(item["id"])
        # 读取图片名字
        for item in data["images"]:
            res[str(item["id"])] = {}
            res[str(item["id"])]["imgName"] = item["file_name"]
            res[str(item["id"])]["w"] = item["width"]
            res[str(item["id"])]["h"] = item["height"]
            res[str(item["id"])]["bbox"] = []
        # 读取bbox
        for item in data["annotations"]:
            resItem = res[str(item["image_id"])]
            t = [classes.index(int(item["category_id"]))]
            bbox = np.array(item["bbox"], dtype="float")
            bbox[:2] += bbox[2:] / 2
            w = resItem["w"]
            h = resItem["h"]
            bbox = (bbox / np.array([w,h,w,h])).astype(float)
            t.extend(bbox.tolist())
            resItem["bbox"].append(t)
        # 输出成txt文件
        valName = []
        for k,v in res.items():
            valName.append(v["imgName"])
            with open(f'{CONST.cocoPath}/labels/val2014/{v["imgName"]}.txt',"w") as f:
                bbox = v["bbox"]
                bbox = [[str(b) for b in item] for item in bbox]
                bbox = [" ".join(item) for item in bbox]
                bbox = '\n'.join(bbox)
                f.write(bbox)
        valName.sort()
        with open(f'{CONST.cocoPath}/val2014.txt',"w") as f:
            f.write('\n'.join(valName))
    # 处理训练集
    res = {}
    with open(f'{CONST.cocoPath}/instances_train2014.json', 'r') as f:
        data = json.load(f)
        classes = []
        # 读取类型
        for item in data["categories"]:
            classes.append(item["id"])
        # 读取图片名字
        for item in data["images"]:
            res[str(item["id"])] = {}
            res[str(item["id"])]["imgName"] = item["file_name"]
            res[str(item["id"])]["w"] = item["width"]
            res[str(item["id"])]["h"] = item["height"]
            res[str(item["id"])]["bbox"] = []
        # 读取bbox
        for item in data["annotations"]:
            resItem = res[str(item["image_id"])]
            t = [classes.index(int(item["category_id"]))]
            bbox = np.array(item["bbox"], dtype="float")
            bbox[:2] += bbox[2:] / 2
            w = resItem["w"]
            h = resItem["h"]
            bbox = (bbox / np.array([w,h,w,h])).astype(float)
            t.extend(bbox.tolist())
            resItem["bbox"].append(t)
        # 输出成txt文件
        valName = []
        for k,v in res.items():
            valName.append(v["imgName"])
            with open(f'{CONST.cocoPath}/labels/train2014/{v["imgName"]}.txt',"w") as f:
                bbox = v["bbox"]
                bbox = [[str(b) for b in item] for item in bbox]
                bbox = [" ".join(item) for item in bbox]
                bbox = '\n'.join(bbox)
                f.write(bbox)
        valName.sort()
        with open(f'{CONST.cocoPath}/train2014.txt',"w") as f:
            f.write('\n'.join(valName))
    print("处理COCO完成")

train_test = 9 # 训练集与测试集比例
train_val = 9 # 训练集与验证集比例
#=========================================================
#           处理原始coco数据集，生成txt文件
#=========================================================
def getImgData(filePath):
    root=ET.parse(CONST.vocPath + "/Annotations/" + filePath).getroot()
    picInfo = {
        "id": filePath[:filePath.index(".xml")],
        "boxes": []
    }
    sizeBox = root.find('size')
    w = int(sizeBox.find("width").text)
    h = int(sizeBox.find("height").text)
    for obj in root.iter('object'):
        # 不使用difficult的数据
        if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
            continue
        cls = obj.find('name').text # 类名
        if cls not in CONST.vocClass:
            continue
        clsType = CONST.vocClass.index(cls)
        xmlbox = obj.find('bndbox')
        box = [int(clsType),int(xmlbox.find("xmin").text) / w, int(xmlbox.find("ymin").text) / h, int(xmlbox.find("xmax").text) / w, int(xmlbox.find("ymax").text) / h]
        picInfo["boxes"].append(box)
        
    return picInfo
def handleVoc():
    # 获取数据集数据
    random.seed(1)
    xmlFile = os.listdir(CONST.vocPath + "/Annotations")
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

    fileNames = []
    for item in val:
        imgName = item["id"] + ".jpg"
        fileNames.append(imgName)
        boxes = np.array(item["boxes"])
        # print(item)
        boxes[:,1:] = np.concatenate([(boxes[:,3:5] + boxes[:,1:3]) / 2, boxes[:,3:5] - boxes[:,1:3]], 1)
        np.savetxt(f'{CONST.vocPath}/labels/val/{imgName}.txt', boxes, fmt="%f")
    fileNames.sort()
    with open(f'{CONST.vocPath}/val2007.txt',"w") as f:
            f.write('\n'.join(fileNames))
    fileNames = []
    for item in train:
        imgName = item["id"] + ".jpg"
        fileNames.append(imgName)
        boxes = np.array(item["boxes"])
        boxes[:,1:] = np.concatenate([(boxes[:,3:5] + boxes[:,1:3]) / 2, boxes[:,3:5] - boxes[:,1:3]], 1)
        np.savetxt(f'{CONST.vocPath}/labels/train/{imgName}.txt', boxes, fmt="%f")
    fileNames.sort()
    with open(f'{CONST.vocPath}/train2007.txt',"w") as f:
            f.write('\n'.join(fileNames))
    print("处理VOC完成")

#=========================================================
#        主程序
#=========================================================   
if __name__ == "__main__":
    handleCOCO()
    # handleVoc()
