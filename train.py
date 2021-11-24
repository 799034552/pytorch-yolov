
import sys
from numpy.core.fromnumeric import size
sys.path.append("./util")
from torch import nn
import torch
from util.const import CONST
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from util.dataload import YOLODataSet, yolo_dataset_collate
from util.model import MyYOLO, getWeight, LastLayer, initialParam
from util.utils import iouOne2One,iou,xywh2xyxy,xyxy2xywh
#==================================#
#           损失函数       
#==================================#
def getLoss(yoloOut, yolo,bboxes):
    BCELoss = nn.BCELoss()
    MSELoss = nn.MSELoss()
    bboxes = torch.cat([bboxes, torch.zeros(bboxes.shape[0],1,device=CONST.device)], 1)
    anchorRelate = torch.tensor(CONST.anchor, device=CONST.device).view(-1,2) / 416
    anchorRelate = torch.cat([torch.zeros_like(anchorRelate), anchorRelate], 1)
    boxesWH = torch.cat([torch.zeros_like(bboxes[:,4:6]), bboxes[:,4:6]], 1)
    for i,item in enumerate(boxesWH):
        bboxes[i][6] = torch.argmax(iou(item, anchorRelate)) # [bs, cls, x,y,w,h,an]
    # print(bboxes)
    loss = 0
    for l,output in enumerate(yoloOut):
        lastLayer = yolo.lastLayers[l]
        ba,c,h,w = output.shape
        output = output.view(ba,len(lastLayer.anchor),-1,h,w).permute(0,1,3,4,2).contiguous()
        b, cls, boxesScaled, an, i, j = buildTarget(bboxes, lastLayer, l)
        tConf = torch.zeros_like(output[..., 4], device=CONST.device)
        xLoss,yLoss,wLoss,hLoss,clsLoss = [0,0,0,0,0]
        if b.shape[0] != 0:
            pr = output[b, an, i, j] # type:torch.Tensor
            tConf[b, an, i, j] = 1
            pr[:,:2] = pr[:,:2].sigmoid()
            xLoss = BCELoss(pr[..., 0], boxesScaled[...,0])
            yLoss = BCELoss(pr[..., 1], boxesScaled[...,1])
            wLoss = MSELoss(pr[..., 2], boxesScaled[...,2]) * 0.5
            hLoss = MSELoss(pr[..., 3], boxesScaled[...,3]) * 0.5
            clsLoss = BCELoss(pr[:,5:].sigmoid(), cls)
        confLoss = BCELoss(output[..., 4].sigmoid(),tConf)
        loss = loss + xLoss + yLoss + wLoss + hLoss + clsLoss + confLoss
    return loss

#==================================#
#        返回这一层的目标框   
#==================================#
def buildTarget(bboxes:torch.Tensor, lastLayer:LastLayer, l):
    corrBox = []
    h,w = lastLayer.shape
    for item in bboxes:
        if item[-1] in CONST.anchorIndex[l]:
            item[-1] = CONST.anchorIndex[l].index(item[-1])
            corrBox.append(item.view(1,-1))
    corrBox = torch.cat(corrBox) if len(corrBox) else torch.Tensor(size=(0,7)).to(CONST.device)
    b = corrBox[:,0].long()
    cl = corrBox[:, 1].long()
    cls = torch.zeros((cl.shape[0], CONST.classNumber), device=CONST.device)
    cls[torch.arange(cl.shape[0]), cl] = 1
    an = corrBox[:,-1].long()
    boxesScaled = corrBox[:,2:6] * torch.tensor([w,h,w,h], device=CONST.device)
    ij = boxesScaled[:,:2].long()
    boxesScaled[:,:2] = boxesScaled[:,:2] - ij
    i = ij[:, 0]
    j = ij[:, 1]
    boxesScaled[:,2:4] = torch.log(boxesScaled[:,2:4] / torch.tensor([w,h], device=CONST.device).view(1,2))
    return b, cls, boxesScaled, an, i, j

#==================================#
#           训练            
#==================================#
def train():
    yolo = MyYOLO() #type: nn.Module
    yolo.apply(initialParam) # 迁移学习
    getWeight(yolo)
    yolo.train()
    yolo.to(CONST.device)
    trainDataSet = YOLODataSet(train=True, type="coco")
    trainDataLoader = DataLoader(trainDataSet, batch_size=CONST.batchSize, num_workers=CONST.num_workers, shuffle=True, pin_memory=True,
                                    drop_last=True,collate_fn=yolo_dataset_collate)
    optimizer       = optim.Adam(yolo.parameters(), 5e-4, weight_decay = 5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    for epoch in range(CONST.epochs):
        with tqdm(total=len(trainDataLoader),postfix=dict,mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{CONST.epochs}')
            s = 0
            for imgs, bboxes in trainDataLoader:
                imgs = imgs.to(CONST.device)
                bboxes = bboxes.to(CONST.device)
                optimizer.zero_grad()
                yoloOut = yolo(imgs)
                loss = getLoss(yoloOut, yolo,bboxes)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{"loss":loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)
                s += 1
                if s == 1000:
                    torch.save(yolo.state_dict(),"weight.pth")
                    s = 0
            lr_scheduler.step()
            pbar.close()
        torch.save(yolo.state_dict(),"weight.pth")

if __name__=='__main__':
    train()