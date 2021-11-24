#--------------------------------------------------------
#               加载数据集
#--------------------------------------------------------
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from const import *
import xml.etree.ElementTree as ET
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
from torchvision import transforms
import warnings
#==================================#
#      读取数据集                        
#==================================#
class YOLODataSet(Dataset):
    def __init__(self, type="coco", train=True) -> None:
        super(YOLODataSet, self).__init__()
        self.type = type
        if train:
            self.dataType = 'train'
            self.transform = TRAIN_TRANSFORMS
        else:
            self.dataType = 'val'
            self.transform = VAL_TRANSFORMS

        if type == "coco":
            with open(CONST.cocoPath + f'/{self.dataType}2014.txt', "r") as file:
                self.imgFiles = file.readlines()
            self.imgFiles =[ item.strip('\n') for item in self.imgFiles]
        elif type == 'voc':
            with open(CONST.vocPath + f'/{self.dataType}2007.txt', "r") as file:
                self.imgFiles = file.readlines()
            self.imgFiles =[ item.strip('\n') for item in self.imgFiles]          

    def __len__(self):
        return len(self.imgFiles)
    
    def __getitem__(self, index):
        fileName = self.imgFiles[index]
        if self.type == 'coco':
            img = Image.open(f"{CONST.cocoPath}/images/{self.dataType}2014/{fileName}").convert("RGB")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boxes = np.loadtxt(f"{CONST.cocoPath}/labels/{self.dataType}2014/{fileName}.txt").reshape((-1, 5))
        elif self.type == 'voc':
            img = Image.open(f"{CONST.vocPath}/JPEGImages/{fileName}").convert("RGB")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boxes = np.loadtxt(f"{CONST.vocPath}/labels/{self.dataType}/{fileName}.txt").reshape((-1, 5))
        img = np.array(img)
        if boxes is None:
            boxes = []
        img, boxes = self.transform((img, boxes))
        # print(fileName)
        return img, boxes


# 绝对值化标签
class AbsoluteLabel():
    def __call__(self, data):
        img, boxes = data
        h, w = img.shape[:2]
        boxes[:,1:] *= np.array([w,h,w,h])
        return img, boxes
# 中点长宽框转换为左上角右下角框
class xywh2xyxy():
    def __call__(self, data):
        img, boxes = data
        boxes[:,1:] = np.concatenate(((boxes[:,1:3] -  boxes[:,3:5] / 2), (boxes[:,1:3] +  boxes[:,3:5] / 2)), axis=1)
        return img, boxes
# 左上角右下角转换为框中点长宽框
class xyxy2xywh():
    def __call__(self, data):
        img, boxes = data
        boxes[:,1:] = np.concatenate((((boxes[:,1:3] +  boxes[:,3:5]) / 2), (boxes[:,3:5] - boxes[:,1:3])), axis=1)
        return img, boxes
# 相对化标签
class RelativeLabel():
    def __call__(self, data):
        img, boxes = data
        h, w = img.shape[:2]
        boxes[:,1:] /= np.array([w,h,w,h])
        return img, boxes
# 应用iaa的图像数据增强类
class imgAug():
    def __init__(self) -> None:
        self.argument = None
    def __call__(self, data):
        img, boxes = data
        bbs = []
        for item in boxes:
            bbs.append(BoundingBox(*item[1:], label=item[0]))
        bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
        img, bbs = self.argument(image = img, bounding_boxes=bbs)
        bbs = bbs.clip_out_of_image()
        for i, item in enumerate(bbs):
            boxes[i,:] = np.array([item.label, item.x1, item.y1, item.x2, item.y2])
        return img, boxes

# 图片长宽比为1
class CenterPlcae(imgAug):
    def __init__(self) -> None:
        super(CenterPlcae, self).__init__()
        self.argument = iaa.Sequential([iaa.PadToAspectRatio(1.0,position="center-center")])

# 图像数据增强
class ImgUp(imgAug):
    def __init__(self) -> None:
        super(ImgUp, self).__init__()
        # 平移缩放、锐化、改变亮度、改变色调、翻转
        self.argument = iaa.Sometimes(0.7, iaa.Sequential([
            iaa.Sometimes(0.8, iaa.Affine(scale=(0.5, 1.2), translate_percent=(-0.2, 0.2))),
            iaa.Sometimes(0.5, iaa.Sharpen((0.0, 0.1))),
            iaa.Sometimes(0.5, iaa.AddToBrightness((-60, 40))),
            iaa.Sometimes(0.5, iaa.AddToHue((-10, 10))),
            iaa.Sometimes(0.3, iaa.Fliplr(1)),
        ]))
# 改变尺寸
class ReSize(imgAug):
    def __init__(self) -> None:
        super(ReSize, self).__init__()
        # 平移缩放、锐化、改变亮度、改变色调、翻转
        self.argument = iaa.Resize((416, 416))
# 转换为tensor
class ToTensor():
    def __call__(self, data):
        img, boxes = data
        img = np.transpose(img / 255, (2, 0, 1))
        img = torch.Tensor(img)
        boxes = torch.Tensor(boxes)
        return img, boxes

#==================================#
#      统一数据格式                     
#==================================#
def yolo_dataset_collate(data):
    images = None
    bboxes = None
    for i,(img, boxes) in enumerate(data):
        images = torch.cat([images, img.unsqueeze(0)]) if images is not None else img.unsqueeze(0)
        t = torch.cat([torch.ones((boxes.shape[0], 1)) * i , boxes], 1)
        bboxes = torch.cat([bboxes, t]) if bboxes is not None else t
    images = images
    return images, bboxes
# 训练集数据增强
TRAIN_TRANSFORMS = transforms.Compose([
    AbsoluteLabel(),
    xywh2xyxy(),
    ImgUp(),
    CenterPlcae(),
    ReSize(),
    RelativeLabel(),
    xyxy2xywh(),
    ToTensor()
])
# 验证集不需要数据增强
VAL_TRANSFORMS = transforms.Compose([
    AbsoluteLabel(),
    xywh2xyxy(),
    CenterPlcae(),
    ReSize(),
    RelativeLabel(),
    xyxy2xywh(),
    ToTensor()
])
trainDataSet = YOLODataSet(train=True)
trainDataLoader = DataLoader(trainDataSet, batch_size=CONST.batchSize, num_workers=CONST.num_workers, shuffle=False, pin_memory=True,
                                drop_last=True,collate_fn=yolo_dataset_collate)
