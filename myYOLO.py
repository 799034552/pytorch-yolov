import PIL
import cv2
import numpy as np
from torch import nn, torch_version
import torchvision
from PIL import Image,ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
from torch.functional import Tensor
from utils import *
from const import *
import math
import random
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
#==================================#
#          简单的卷积层                      
#==================================#
class Conv(nn.Module):
    def __init__(self, inputC, outputC, keralSize, stride = 1, padding = "same") -> None:
        super(Conv, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(inputC, outputC, keralSize, stride, padding, bias=False),
            nn.BatchNorm2d(outputC),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.m(x)
#==================================#
#             残差块                      
#==================================#
class Residual(nn.Module):
    def __init__(self, inputC) -> None:
        super(Residual, self).__init__()
        tempC = inputC // 2
        self.m = nn.Sequential(
            Conv(inputC, tempC, 1, 1, 0),
            Conv(tempC, inputC, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.m(x)
#==================================#
#           darknet53                       
#==================================#
class Darknet53(nn.Module):
    def __init__(self) -> None:
        super(Darknet53, self).__init__()
        # 定义darknet53的层数
        self.layoutNumber = [1, 2, 8, 8, 4]
        self.layerA = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            self.MultiResidual(32, 64, 1),
            self.MultiResidual(64, 128, 2),
            self.MultiResidual(128, 256, 8)
        )
        self.layerB = self.MultiResidual(256, 512, 8)
        self.layerC = self.MultiResidual(512, 1024, 4)

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out1 = self.layerA(x)
        out2 = self.layerB(out1)
        out3 = self.layerC(out2)
        return out1, out2, out3

    # 多层的残差网络
    def MultiResidual(self, inputC, outputC, count):
        t = [Conv(inputC, outputC, 3, 2, 1) if i == 0 else Residual(outputC) for i in range(count + 1)]
        return nn.Sequential(*t)

#==================================#
#           convSet                    
#==================================#
class convSet(nn.Module):
    def __init__(self, inputC, outputC, midC) -> None:
        super(convSet, self).__init__()
        self.m = nn.Sequential(
            Conv(inputC, outputC, 1),
            Conv(outputC, midC, 3),
            Conv(midC, outputC, 1),
            Conv(outputC, midC, 3),
            Conv(midC, outputC, 1),
        )
    def forward(self, x):
        return self.m(x)

#==================================#
#           lastLayer                   
#==================================#
class LastLayer(nn.Module):
    def __init__(self, inputC, outputC) -> None:
        super(LastLayer, self).__init__()
        self.m = nn.Sequential(
            Conv(inputC, inputC * 2, 3),
            nn.Conv2d(inputC * 2, outputC, 1)
        )
    def forward(self, x):
        return self.m(x)
#==================================#
#           定义yolo模型                        
#==================================#
class MyYOLO(nn.Module):
    def __init__(self) -> None:
        super(MyYOLO, self).__init__()
        # 得到 1024*13*13
        self.darknet53 = Darknet53()
        # 得到 512*13*13
        self.convSet1 = convSet(1024, 512, 1024)
        # 得到 256*26*26, 但是后面要和另一层的输出合起来，得到的应该是 (512+256)*26*26
        self.layerA = nn.Sequential(
            Conv(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # 得到 256*26*26
        self.convSet2 = convSet(256 + 512, 256, 512)
        
        # 得到 128*52*52, 但是后面要和另一层的输出合起来，得到的应该是 (128+256)*52*52
        self.layerB = nn.Sequential(
            Conv(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # 得到 256*26*26
        self.convSet3 = convSet(128 + 256, 128, 256)

        # 得到 75*13*13
        self.predict1 = LastLayer(512, CONST.anchorNumber * (5 + CONST.classNumber))
        # 得到 75*26*26
        self.predict2 = LastLayer(256, CONST.anchorNumber * (5 + CONST.classNumber))
        # 得到 75*52*52
        self.predict3= LastLayer(128, CONST.anchorNumber * (5 + CONST.classNumber))
    def forward(self, x):
        x1, x2, x3 = self.darknet53(x)
        convOut1 = self.convSet1(x3)
        out1 = self.predict1(convOut1)
        layerOut = self.layerA(convOut1)
        additon = torch.cat([layerOut, x2], 1)
        convOut2 = self.convSet2(additon)
        out2 = self.predict2(convOut2)
        layerOut = self.layerB(convOut2)
        additon = torch.cat([layerOut, x1], 1)
        convOut3 = self.convSet3(additon)
        out3 = self.predict3(convOut3)

        return out1, out2, out3

#==================================#
# 迁移学习                        
#==================================#
def getWeight(yolo):
    weightData = torch.load("yolo_weights.pth", map_location="cuda")
    myWeightData = yolo.state_dict()
    keys = list(yolo.state_dict().keys())
    keys = np.concatenate([keys[:342], keys[414:422], keys[342:378], keys[422:430],keys[378:414],keys[430:438]])
    i = 0
    for k, v in weightData.items():
        if keys[i].find("num_batches_tracked") != -1:
            i+=1
        myWeightData[keys[i]].copy_(v)
        i+=1
    yolo.eval()


#==================================#
# 处理先验证框，将先验框映射到与输入尺寸同样的大小                        
#==================================#
def handleBox(input, anchorPlace = [[6,7,8],[3,4,5],[0,1,2]]):
    outputs = []
    for i, res in enumerate(input):
        # 读取值
        (batchSize, channel, height, width) = res.shape
        ih, iw = CONST.inputShape
        res = res.view(batchSize, CONST.anchorNumber, -1, height, width).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(res[..., 0])
        y = torch.sigmoid(res[..., 1])
        w = res[..., 2]
        h = res[..., 3]
        p = torch.sigmoid(res[..., 4])
        classesP = torch.sigmoid(res[..., 5:])

        scaleX = iw / width
        scaleY = ih / height
        anchorBoxs = torch.tensor(CONST.anchor)[anchorPlace[i]] / torch.tensor([scaleX, scaleY])

        # 生成先验框的位置
        gridX = torch.ones_like(x).cuda() * torch.linspace(0, width - 1, width).cuda()
        gridY = torch.ones_like(y).cuda() * torch.linspace(0, height - 1, height).view(-1, 1).cuda()
        gridW = torch.ones_like(w).cuda() * anchorBoxs[:,0].view(1,3,1,1).cuda()
        gridH = torch.ones_like(h).cuda() * anchorBoxs[:,1].view(1,3,1,1).cuda()
        # 得到先验框相对于图像的位置
        x = x + gridX
        y = y + gridY
        w = torch.exp(w) * gridW
        h = torch.exp(h) * gridH
        x.unsqueeze_(-1)
        y.unsqueeze_(-1)
        w.unsqueeze_(-1)
        h.unsqueeze_(-1)
        # 归一化为相对坐标
        place = torch.cat([x, y, w, h], -1) / torch.Tensor([width, height, width, height]).cuda()
        out = torch.cat([place.view(batchSize, -1, 4), p.view(batchSize, -1, 1), classesP.view(batchSize, -1, classesP.shape[-1])], -1)
        outputs.append(out)
    # 返回的数据大小为(batchSize, 10647， 85)
    return torch.concat(outputs,1)
#==================================#
#        一个框与多个框的交并比                        
#==================================#
def iou(box1: torch.Tensor, box2:torch.Tensor, isleftT2rightD = True) -> torch.Tensor:
    # box1 的shape为(1, 4), box2的shape为(None, 4)
    # 防止输入错误
    box1 = box1.repeat((box2.shape[0], 1))
    if not isleftT2rightD:
        box1 = torch.concat([box1[:,:2] - box1[:,2:4] / 2, box1[:,:2] + box1[:,2:4] / 2], 1).cuda()
        box2 = torch.concat([box2[:,:2] - box2[:,2:4] / 2, box2[:,:2] + box2[:,2:4] / 2], 1).cuda()
    # 交集左上角的点
    lu = torch.max(box1[:, :2], box2[:, :2])
    # 交集右下角的点
    rd = torch.min(box1[:, 2:], box2[:, 2:])
    rectsN = rd - lu
    rectsN[rectsN < 0] = 0#没有重叠区域设置为0
    rectsN = rectsN[:,0] * rectsN[:,1]
    rectsU = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]) + (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    return rectsN / (rectsU - rectsN)

#==================================#
#           非极大值抑制                        
#==================================#
def nms(box: torch.Tensor = None, score: torch.Tensor = None,threshold: float = 0.3) -> None:
    _, sortIndex =  score.sort(0, descending = True)
    res = []
    while sortIndex.size(0):
        if sortIndex.size(0) == 1:
            res.append(sortIndex[0].item())
            break
        res.append(sortIndex[0].item())
        iou = iou(box[sortIndex[0]], box[sortIndex[1:]])
        sortIndex = sortIndex[1:][iou < threshold]
    return  res
    # 交并比

#==================================#
#       非极大值抑制                    
#==================================#
def notMaxSuppression(inputVal, confThres = 0.5):
    # 化为左上角+右下角坐标
    box_corner = inputVal.new(inputVal.shape)
    box_corner[:, :, 0] = inputVal[:, :, 0] - inputVal[:, :, 2] / 2
    box_corner[:, :, 1] = inputVal[:, :, 1] - inputVal[:, :, 3] / 2
    box_corner[:, :, 2] = inputVal[:, :, 0] + inputVal[:, :, 2] / 2
    box_corner[:, :, 3] = inputVal[:, :, 1] + inputVal[:, :, 3] / 2
    inputVal[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in inputVal]
    for i, prediction in enumerate(inputVal):
        # 置信度与对应的类型
        classP, classType = torch.max(prediction[:, 5:], 1, keepdim=True)
        # 利用置信度进行第一轮筛选
        confMask = (prediction[...,4] * classP[...,0] >= confThres)
        
        prediction = prediction[confMask]
        classP = classP[confMask]
        classType = classType[confMask]
 
        if not prediction.shape[0]:
            continue
        # 整合数据
        prediction = torch.cat([prediction[:,:5], classP, classType], 1)
        uniqueClass = prediction[:, -1].unique()
        # 对每一类分别进行非极大值抑制
        for uClassType in uniqueClass:
            tPrediction = prediction[prediction[:, -1] == uClassType]
            # if tPrediction.size(0) == 1:
            #     continue
            res = nms(tPrediction[:,:4], tPrediction[:,4] * tPrediction[:,5], threshold=0.3)
            # res = torchvision.ops.nms(tPrediction[:,:4], tPrediction[:,4] * tPrediction[:,5], 0.3) 这是torch自带的的nms
            tPrediction = tPrediction[res]
            output[i] = tPrediction if output[i] is None else torch.cat([output[i], tPrediction])
    return output

#==================================#
#       原图坐标映射                    
#==================================#
def refer(input, imageSize, netInputSize = (416,416), isHold = False):
    if not isHold:
        for batchNum, val in  enumerate(input):
            val[...,:4] = val[...,:4] * torch.Tensor(imageSize).repeat((1,2)).cuda()
            input[batchNum] = val
    return input

#==================================#
#       绘制半透明框                  
#==================================#
def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))
#==================================#
#           预测                  
#==================================#
def predict(imgL, isShowRaw=False):
    # 读取并处理图片
    if isinstance(imgL, str):
        img, rawImg = handlePic(imgL)
    else:
        img = imgL
        if isShowRaw:
            Image.fromarray(np.transpose(np.uint8(img[0]*255), [1, 2, 0])).show()
            isShowRaw = False
    if isShowRaw:
        Image.fromarray(np.transpose(np.uint8(img[0]*255), [1, 2, 0])).show() # 显示输入图片
        rawImg.show()
    with torch.no_grad():
        img = torch.from_numpy(img)
        img = img.float()
        img = img.cuda()
        yolo = MyYOLO().cuda() # type: torch.nn.Module
        # 迁移学习
        getWeight(yolo)
        # 预测图片
        output = yolo(img) # 返回的数据大小为[(batchSize, 255,13,13), (batchSize, 255,13,13)]
        output = handleBox(output) # 处理先验框 返回的数据大小为(batchSize, 10647， 85)
        output = notMaxSuppression(output) # 非极大值抑制
        print(f"抑制后的结果:{output[0].shape}")
        output = refer(output, rawImg.size) # 将图片映射到原图坐标
        output = output[0].cpu()
        if len(output) == 0:
            print("没有找到特征")
            exit()
        oLable = np.array(output[:,6], dtype="int32")
        oBox = np.array(output[:,:4], dtype="int32")
        oP = np.array(output[:,4] * output[:, 5], dtype="float")
        # 开始绘图
        rawImg = rawImg.convert("RGBA")
        # 矩形厚度
        thickness = max(2, rawImg.size[0] // 200)
        # 每一种类型对应的颜色
        colors = getColor(CONST.classNumber)
        # 字体大小
        font = ImageFont.truetype(font = "arial.ttf", size=np.floor(3e-2 * rawImg.size[1] + 0.5).astype('int32'))
        draw = ImageDraw.Draw(rawImg)
        for j, item in enumerate(oBox):
            color = colors[oLable[j]]
            for i in range(thickness):
                x1 = max(0, item[0] - i)
                y1 = max(0, item[1] - i)
                x2 = min(rawImg.size[0], item[2] + i)
                y2 = min(rawImg.size[1], item[3] + i)
                drawRect(rawImg, [x1, y1, x2, y2], outline=color)
            x1 = x1
            y1 = max(0, item[1] - rawImg.size[0] // 35)
            x2 = min(rawImg.size[0], x1 + rawImg.size[0] // 5)
            y2 = item[1]
            drawRect(rawImg, [x1, y1, x2, y2], fill=color)
            draw.text((x1, y1), str(CONST.classes[oLable[j]])+ " " + str(int(oP[j] * 100) / 100), fill=(0, 0, 0), font=font)
        rawImg.show()

#==================================#
#           初始化参数        
#==================================#
def initialParam(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, std=1e-3)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

#==================================#
#           损失函数       
#==================================#
class YOLOLoss(nn.Module):
    def __init__(self) -> None:
        super(YOLOLoss, self).__init__()
        self.anchorPlace = [[6,7,8],[3,4,5],[0,1,2]]
        self.threshold = 0.5
    
    def forward(self, l, input, targets=None):
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        # 获取长宽
        (bs, c, h, w) = input.shape
        scaleX = CONST.inputShape[0] / w
        scaleY = CONST.inputShape[1] / h
        anchorBoxs = torch.tensor(CONST.anchor) / torch.tensor([scaleX, scaleY])
        res = input.view(bs, CONST.anchorNumber, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
        resX = torch.sigmoid(res[..., 0]).cuda()
        resY = torch.sigmoid(res[..., 1]).cuda()
        resW = res[..., 2]
        resH = res[..., 3]
        p = torch.sigmoid(res[..., 4]).cuda()
        classesP = torch.sigmoid(res[..., 5:]).cuda()
        gridX = torch.ones_like(resX).cuda() * torch.linspace(0, w - 1, w).cuda()
        gridY = torch.ones_like(resY).cuda() * torch.linspace(0, h - 1, h).view(-1, 1).cuda()
        gridW = torch.ones_like(resW).cuda() * anchorBoxs[self.anchorPlace[l]][:,0].view(1,3,1,1).cuda()
        gridH = torch.ones_like(resH).cuda() * anchorBoxs[self.anchorPlace[l]][:,1].view(1,3,1,1).cuda()
        # 得到先验框相对于图像的位置
        resXByAnchor = resX + gridX
        resYByAnchor = resY + gridY
        resWByAnchor = torch.exp(resW) * gridW
        resHByAnchor = torch.exp(resH) * gridH
        resXByAnchor.unsqueeze_(-1)
        resYByAnchor.unsqueeze_(-1)
        resWByAnchor.unsqueeze_(-1)
        resHByAnchor.unsqueeze_(-1)
        predictBox = torch.cat([resXByAnchor, resYByAnchor, resWByAnchor, resHByAnchor], -1)

        # 获得网络应有的预测结果
        noObjMask = torch.ones(bs, len(self.anchorPlace[l]), h, w).cuda()
        # 让网络更关注小物体
        smallWeight = torch.ones(bs, len(self.anchorPlace[l]), h, w).cuda()
        trueY = torch.zeros(bs, len(self.anchorPlace[l]), h, w, len(CONST.classes) + 5).cuda()
        for b in range(bs):
            # 抽离出一张图片的真实框
            batchTarget = torch.Tensor(targets[b])
            batchPredictBox = predictBox[b] # (3 * h * w * 4)
            batchPredictBox = batchPredictBox.view(-1, 4)
            
        
            batchTarget[:,[0,2]] *= w
            batchTarget[:,[1,3]] *= h
            # 处理真实框只剩下长宽
            boxTarget = torch.cat([torch.zeros(batchTarget.shape[0], 2), batchTarget[:,[2,3]]], 1).cuda()
            boxAnchor = torch.cat([torch.zeros(anchorBoxs.shape[0], 2), anchorBoxs], 1).cuda()
            
            # 得到每个真实框最大交并比的先验框
            maxOne = [torch.argmax(iou(trueBox, boxAnchor,isleftT2rightD= False)).item() for trueBox in boxTarget]
            # 对于每个真实框，需要计算出神经网络应该输出的数据，以及预测出来应该判断为对的框
            for maxI, oneTrueBox in enumerate(batchTarget):
                # 如果不是对应的框就放弃，说明不是这个层负责预测的
                if maxOne[maxI] not in self.anchorPlace[l]:
                    continue
                k = self.anchorPlace[l].index(maxOne[maxI])
                i,j = oneTrueBox[0:2].numpy().astype('int32').tolist() # 真实框属于哪个网格
                truetype = int(oneTrueBox[4].item())
                noObjMask[b, k, j, i] = 0 # 标记该位置有物体
                # 获得神经网络应该输出的数据
                trueY[b, k, j, i, 0] = oneTrueBox[0] - i
                trueY[b, k, j, i, 1] = oneTrueBox[1] - j
                trueY[b, k, j, i, 2] = math.log(oneTrueBox[2] / anchorBoxs[maxOne[maxI]][0] )
                trueY[b, k, j, i, 3] = math.log(oneTrueBox[3] / anchorBoxs[maxOne[maxI]][1] )
                trueY[b, k, j, i, 4] = 1
                trueY[b, k, j, i, 5 + truetype] = 1
                smallWeight[b, k, j, i] = oneTrueBox[2] * oneTrueBox[3] / w / h
                # 预测值与真实值的交比并大于0.5的话就认为预测正确
                iouRes = iou(oneTrueBox[:4], batchPredictBox, isleftT2rightD=False)
                iouRes = iouRes.view(predictBox[b].shape[:3])
                noObjMask[b][iouRes > self.threshold] = 0

        smallWeight = 2 - smallWeight
        lossX = torch.sum(nn.BCELoss()(resX, trueY[..., 0]) * trueY[..., 4] * smallWeight)
        lossY = torch.sum(nn.BCELoss()(resY, trueY[..., 1]) * trueY[..., 4] * smallWeight)
        lossW = torch.sum(nn.MSELoss()(resW, trueY[..., 2]) * trueY[..., 4] * smallWeight * 0.5)
        lossH = torch.sum(nn.MSELoss()(resH, trueY[..., 3]) * trueY[..., 4] * smallWeight * 0.5)
        lossP = torch.sum(nn.BCELoss()(p, trueY[..., 4]) * trueY[..., 4]) +\
                torch.sum(nn.BCELoss()(p, trueY[..., 4]) * noObjMask)
        if torch.sum(trueY[..., 4]).item() != 0:
            lossClassP = torch.sum(nn.BCELoss()(classesP[trueY[..., 4] == 1], trueY[..., 5:][trueY[..., 4] == 1]))
        else:
            lossClassP = torch.tensor(0).cuda()
        loss = lossX + lossY + lossW + lossH + lossP + lossClassP
        return loss, torch.max(torch.sum(trueY[..., 4]), torch.tensor(1).cuda())

            



                




            


            

                    

#==================================#
#           训练            
#==================================#
def train():
    yolo = MyYOLO() #type: nn.Module
    yolo.apply(initialParam) # 迁移学习
    yolo.train()
    # 启用多GPU并加速训练
    model_train = torch.nn.DataParallel(yolo)
    torch.backends.cudnn.benchmark = True
    model_train = model_train.cuda()


#==================================#
#           主函数               
#==================================#
if __name__=='__main__':
    # predict("./img/street.jpg", isShowRaw=False)
    # train()
    # 读取数据集
    yoloLoss = YOLOLoss()
    yolo = MyYOLO()
    getWeight(yolo)
    yolo = yolo.train()
    yolo = yolo.cuda()
    trainData = YOLODataSet(train=True)
    valData = YOLODataSet(train=False)
    trainDataLoader = DataLoader(trainData, batch_size=CONST.batchSize, num_workers=CONST.num_workers, shuffle=False, pin_memory=True,
                                    drop_last=True,collate_fn=yolo_dataset_collate)
    valDataLoader = DataLoader(valData, batch_size=CONST.batchSize, num_workers=CONST.num_workers, shuffle=False,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
    optimizer       = optim.Adam(yolo.parameters(), 1e-3, weight_decay = 5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    # img, box = trainData.__getitem__(0)
    # predict(img)
    # Image.fromarray(np.transpose(np.uint8(img*255), [1, 2, 0])).show() # 显示输入图片
    # epoch = len(trainDataLoader) // CONST.batchSize
    # 先不冻结训练
    batchNum = len(trainDataLoader) // CONST.batchSize
    testLoss = []
    valLoss = []
    for epoch in range(CONST.epochs):
        # 训练集循环
        with tqdm(total=batchNum,postfix=dict,mininterval=0.3, bar_format = '{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') as pbar:
            for batchCount, (imgs, boxes) in enumerate(trainDataLoader):
                pbar.set_description(f'Epoch {epoch + 1}/{CONST.epochs}')
                # Image.fromarray(np.transpose(np.uint8(imgs[0]*255), [1, 2, 0])).show()
                imgs = imgs.cuda()
                optimizer.zero_grad()
                output =  yolo(imgs)
                lossVal = 0
                num = 0
                for l, o in enumerate(output):
                    loss, n = yoloLoss(l, o, boxes)
                    lossVal += loss
                    num += n
                lossVal /= num
                testLoss.append(lossVal)
                
                lossVal.backward()
                optimizer.step()
                pbar.set_postfix(**{"loss":lossVal.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)
            pbar.close()
        torch.save(yolo.state_dict(),"weight.pth")

        # 验证集循环
        with torch.no_grad():
            with tqdm(total=batchNum,postfix=dict,mininterval=0.3, bar_format = '{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') as pbar:
                for batchCount, (imgs, boxes) in enumerate(trainDataLoader):
                    optimizer.zero_grad()
                    pbar.set_description(f'Epoch {epoch + 1}/{CONST.epochs}')
                    # Image.fromarray(np.transpose(np.uint8(imgs[0]*255), [1, 2, 0])).show()
                    imgs = imgs.cuda()
                    optimizer.zero_grad()
                    output =  yolo(imgs)
                    lossVal = 0
                    num = 0
                    for l, o in enumerate(output):
                        loss, n = yoloLoss(l, o, boxes)
                        lossVal += loss
                        num += n
                    lossVal /= num
                    valLoss.append(lossVal)
                    pbar.set_postfix(**{"loss":lossVal.item(), "lr": optimizer.param_groups[0]['lr']})
                    pbar.update(1)
        lr_scheduler.step()




