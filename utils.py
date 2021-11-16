import PIL
import cv2
import numpy as np
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import colorsys
from const import *
import os
import random
import xml.etree.ElementTree as ET
import json
from PIL import Image,ImageDraw, ImageFont
#==================================#
#      读取数据集                        
#==================================#
class YOLODataSet(Dataset):
    def __init__(self, train) -> None:
        super(YOLODataSet, self).__init__()
        with open('./data.json','r') as load_f:
            data = json.load(load_f)
        if train:
            data = data["train"]
        else:
            data = data["val"]
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        boxes = self.data[index]["boxes"]
        img = Image.open(CONST.imgPath + "/" + data["id"] + ".jpg").convert("RGB")
        iw, ih = img.size
        h, w = CONST.inputShape
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # 图片多余部分加灰条
        img       = img.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(img, (dx, dy))
        image_data  = np.array(new_image, np.float32)

        # 对真实框调整
        boxes = np.array(boxes)
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
        boxes[:, 0:2][boxes[:, 0:2]<0] = 0
        boxes[:, 2][boxes[:, 2]>w] = w
        boxes[:, 3][boxes[:, 3]>h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w>1, box_h>1)] # 丢弃没有用的框
        # boxes是左上角，右下角结构
        image_data = np.transpose(image_data / 255, (2, 0, 1))
        boxes = boxes / np.array([w, h, w, h])
        boxes = np.concatenate([boxes, np.array(self.data[index]["clsTypes"]).reshape(-1, 1)],1)
        return image_data, boxes
#==================================#
#      统一数据格式                     
#==================================#
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.Tensor(np.array(images))
    return images, bboxes




    

#==================================#
#      将图片变为合适的大小                        
#==================================#
def resizeImg(img, wantShape, isHold):
    print(wantShape)
    iw, ih = img.size
    w, h = wantShape
    if isHold:
        scale = min(w / iw, h / ih)
        bw = int(iw * scale)
        bh = int(ih * scale)
        img = img.resize((bw, bh))
        new_image = Image.new('RGB', wantShape, (128,128,128))
        new_image.paste(img, ((w-bw)//2, (h-bh)//2))
    else:
        new_image = img.resize(wantShape, Image.BICUBIC)
    return new_image
#==================================#
#          读取并处理图片                        
#==================================#
def handlePic(picName):
    img = Image.open(picName).convert("RGB")
    img1 = resizeImg(img, CONST.inputShape, isHold=False)
    img1 = np.array(img1)
    img1 = np.expand_dims(np.transpose(img1 / 255, (2, 0, 1)), 0)
    return img1, img
#==================================#
#          获取种类颜色                      
#==================================#
def getColor(number):
    random.seed(1)
    hsvColor = [(i / number, 1, 1) for i in range(number)]
    rgbColor = [colorsys.hsv_to_rgb(*x) for x in hsvColor]
    rgbColor = [(int(x[0]*255),int(x[1]*255),int(x[2]*255), 180) for x in rgbColor]
    random.shuffle(rgbColor)
    return rgbColor