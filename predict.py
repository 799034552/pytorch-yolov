import sys
sys.path.append("./util")
from imgaug.augmentables import bbs
import numpy as np
from PIL import Image
import torch
from util.const import CONST
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import imgaug.augmenters as iaa
from util.model import MyYOLO, getWeight
from util.utils import handleBox,notMaxSuppression

#==================================#
#       原图坐标映射                    
#==================================#
def refer(input, imageSize, netInputSize = (416,416), isHold = False):
    if not isHold:
        for batchNum, val in  enumerate(input):
            val[...,:4] = val[...,:4] * torch.Tensor(imageSize).repeat((1,2)).to(CONST.device)
            input[batchNum] = val
    return input

#==================================#
#           展示图片 
#==================================#
def showPic(output, rawImg):
    output = output.to("cpu")
    oLable = np.array(output[:,6], dtype="int32")
    oBox = np.array(output[:,:4], dtype="int32")
    oP = np.array(output[:,4] * output[:, 5], dtype="float")
    boxes = []
    for i, box in enumerate(oBox):
        boxes.append(BoundingBox(*box.tolist(),label=CONST.classes[oLable[i]] + "  " +  str(int(oP[i] * 100) / 100)))
    rawImg = np.array(rawImg)
    bbs = BoundingBoxesOnImage(boxes, shape=rawImg.shape)
    Image.fromarray(bbs.draw_on_image(rawImg, size=4, alpha=0.9)).show()
#==================================#
#           预测      
#==================================#
def predict(img, isShow=True):
    rawImg = Image.open(img).convert("RGB")
    img = np.array(rawImg)
    img = iaa.Sequential([
        iaa.Resize((416,416))
    ])(image=img)
    img = np.transpose(img / 255, (2, 0, 1))
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).type(torch.float).to(CONST.device)
        yolo = MyYOLO().to(CONST.device) # type: torch.nn.Module
        # 迁移学习
        getWeight(yolo)
        # 预测图片
        output = yolo(img) # 返回的数据大小为[(batchSize, 255,13,13), (batchSize, 255,13,13)]
        output = handleBox(output, yolo) # 处理先验框 返回的数据大小为(batchSize, 10647， 85)
        output = notMaxSuppression(output) # 非极大值抑制

        print(f"抑制后的结果:{output[0].shape}")
        output = refer(output, rawImg.size) # 将图片映射到原图坐标
        output = output[0].to("cpu")
        if len(output) == 0:
            print("没有找到特征")
            exit()
        if isShow:
            showPic(output, rawImg)
        return output

#==================================#
#           主函数               
#==================================#
if __name__=='__main__':
    predict("./data/testImg/street.jpg")