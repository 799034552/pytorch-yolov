import sys
sys.path.append("./util")
import numpy as np
from torch import nn
import torch
from util.const import CONST
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.dataload import YOLODataSet, yolo_dataset_collate
from terminaltables import AsciiTable
from util.model import MyYOLO, getWeight,initialParam
from util.utils import handleBox,notMaxSuppression,iou

# 获取正确的框框
def getTrueBox(outputs, bboxes):
    res = []
    for i,output in enumerate(outputs):
        # 对于一张图
        if output is None: # 没有预测框就跳过
            continue
        preBoxes = output[:,:4]
        preLabels = output[:,6]
        preConf = output[:,4] * output[:,5]
        targetBboxes = bboxes[bboxes[:,0] == i]
        targetBoxes = targetBboxes[:,2:]
        targetLabels = targetBboxes[:,1]
        detectedBox = []
        isCor = torch.zeros_like(preLabels)
        for j, preBox in enumerate(preBoxes):
            # 对于一个框
            if (len(detectedBox) == len(targetLabels)):
                break
            # print(iou(preBox, targetBoxes, isleftT2rightD=True))
            iout, maxI = torch.max(iou(preBox, targetBoxes, isleftT2rightD=True), 0)
            if iout > CONST.valIOUTher and maxI not in detectedBox and preLabels[j] == targetLabels[maxI]:
                isCor[j] = 1
                detectedBox.append(maxI)
        res.append([isCor, preConf, preLabels])
    return res

#==================================#
#           计算模型参数               
#==================================#
def calMap(isCor, preConf, preLabels, targetLabels):
    sI = np.argsort(-preConf)
    isCor = isCor[sI]
    preConf = preConf[sI]
    preLabels = preLabels[sI]
    uClasses = np.unique(targetLabels)
    R = []
    P = []
    AP = []
    for oneCls in uClasses:
        sI = preLabels == oneCls
        isCorOneCls = isCor[sI]

        targetLabelsOneCls = targetLabels[targetLabels == oneCls]
        tarTrueC = targetLabelsOneCls.size # 目标框为该类的数量
        preTrueC = isCorOneCls.size # 预测框为该类的数量

        if preTrueC == 0:
            R.append(0)
            P.append(0)
            AP.append(0)
            continue
        tpC = isCorOneCls.cumsum()
        fpC = (1 - isCorOneCls).cumsum()

        r = tpC / tarTrueC
        p = tpC / (tpC + fpC)
        R.append(r[-1])
        P.append(p[-1])
        # 在前面添加是往前取矩形，在后面添加是让召回率可以达到1
        r = np.concatenate(([0.0], r, [1.0]))
        p = np.concatenate(([0.0], p, [0.0]))
        # 保证p单调递减
        for i in range(p.size - 1, 0, -1):
            p[i - 1] = max(p[i], p[i - 1])
        # 删除重复项
        i = np.where(r[1:] != r[:-1])[0]
        ap = np.sum((r[i+1] - r[i]) * p[i+1])
        AP.append(ap)
    return R,P,AP,uClasses
        
#==================================#
#           show MP            
#==================================#
def showMap(R,P,AP,uClasses):
    res = [["class","AP", "R", "P"]]
    for i,_ in enumerate(uClasses):
        res.append([CONST.classes[int(uClasses[i])], "%.4f" % AP[i], "%.4f" % R[i], "%.4f" % P[i]])
    res.append([])
    res.append(["MAP", "%.4f" % np.average(AP)])
    print(AsciiTable(res).table)

#==================================#
#           验证          
#==================================#
def valid():
    yolo = MyYOLO() # type: nn.Module
    getWeight(yolo)
    yolo.eval()
    yolo.to(CONST.device)
    valDataSet = YOLODataSet(train=False, type="coco")
    valDataLoader = DataLoader(valDataSet, batch_size=CONST.batchSize, num_workers=CONST.num_workers, shuffle=False, pin_memory=True,
                                    drop_last=True,collate_fn=yolo_dataset_collate)
    corBox = []
    targetLabels = []
    with torch.no_grad():
        for imgs, bboxes in tqdm(valDataLoader, desc="Validating"):
            imgs = imgs.to(CONST.device)
            bboxes = bboxes.to(CONST.device) # 输入的数据为[picNumber,cls,x,y,w,h]
            output = yolo(imgs)
            output = handleBox(output, yolo) # 处理先验框 返回的数据大小为(batchSize, 10647， 85)
            output = notMaxSuppression(output) # 非极大值抑制, 返回的数据为batchSize[x,y,x,y,conf,cls]
            # print(f"抑制后的结果:{len(output), [x.shape if x is not None else None for x in output]}")
            bboxes[:,2:] = torch.cat([bboxes[:,2:4] - bboxes[:,4:6] / 2, bboxes[:,2:4] + bboxes[:,4:6] / 2], 1) #转换为xyxy
            corBox.extend(getTrueBox(output, bboxes))
            targetLabels.append(bboxes[:,1])
    if len(corBox) == 0:
        print("没有任何输出")
        exit()
    isCor, preConf, preLabels = [torch.cat(x, 0).cpu().numpy() for x in zip(*corBox)]
    targetLabels = torch.cat(targetLabels, 0).cpu().numpy()
    R,P,AP,uClasses = calMap(isCor, preConf, preLabels, targetLabels)
    showMap(R,P,AP,uClasses)

#==================================#
#           主函数               
#==================================#
if __name__=='__main__':
    valid()
