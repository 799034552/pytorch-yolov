
from imgaug.augmentables import bbs
import numpy as np
from torch import nn
import torch
from torch.functional import Tensor
from utils import *
from const import *
import math
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
    def __init__(self, inputC, outputC, anchor=None) -> None:
        super(LastLayer, self).__init__()
        self.grid = None
        self.anchor = np.array(anchor)
        self.anchorScaled = []
        self.stride = 1
        self.shape = None
        self.m = nn.Sequential(
            Conv(inputC, inputC * 2, 3),
            nn.Conv2d(inputC * 2, outputC, 1)
        )
    def forward(self, x):
        o = self.m(x)
        if self.grid is None:
            self._createGrid(o.shape)
        return o
    def _createGrid(self, shape):
        b,c,h,w = shape
        self.shape = (h, w)
        self.stride = CONST.inputShape[0] / h
        self.anchorScaled = torch.tensor(self.anchor / self.stride, device=CONST.device)
        grid = torch.ones((b,len(self.anchor),h,w,4),device=CONST.device)
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid[...,0] *= gridX.to(CONST.device).unsqueeze(0)
        grid[...,1] *= gridY.to(CONST.device).unsqueeze(0)
        grid[...,2] *= self.anchorScaled[:,0].view(1,len(self.anchor),1,1)
        grid[...,3] *= self.anchorScaled[:,1].view(1,len(self.anchor),1,1)
        self.grid = grid
        
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
        self.predict1 = LastLayer(512, CONST.anchorNumber * (5 + CONST.classNumber), anchor=CONST.anchor[0])
        # 得到 75*26*26
        self.predict2 = LastLayer(256, CONST.anchorNumber * (5 + CONST.classNumber), anchor=CONST.anchor[1])
        # 得到 75*52*52
        self.predict3= LastLayer(128, CONST.anchorNumber * (5 + CONST.classNumber), anchor=CONST.anchor[2])
        self.lastLayers = [self.predict1, self.predict2, self.predict3]
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
    yolo.apply(initialParam)
    weightData = torch.load("yolo_weights.pth", map_location="cuda")
    # yolo.load_state_dict(weightData)
    # return
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