import torch
from const import CONST

#==================================#
# 处理先验证框，将先验框映射到与输入尺寸同样的大小                        
#==================================#
def handleBox(input, yolo):
    outputs = []
    lastLayers = yolo.lastLayers
    for i, lasterLayer in enumerate(lastLayers):
        b, c, h, w = input[i].shape
        res = input[i].view(b, len(lasterLayer.anchor), -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
        res[...,[0,1,4]] = res[...,[0,1,4]].sigmoid()
        res[...,5:] = res[...,5:].sigmoid()
        res[...,:2] = res[...,:2] + lasterLayer.grid[..., :2]
        res[...,2:4] = torch.exp(res[...,2:4]) * lasterLayer.grid[..., 2:4]
        res[...,:4] = res[...,:4] / torch.Tensor([w,h,w,h]).to(CONST.device)
        res = res.view(b,-1,5+CONST.classNumber)
        outputs.append(res)
    return torch.cat(outputs,1)

#==================================#
#        一个框与多个框的交并比                        
#==================================#
def iou(box1: torch.Tensor, box2:torch.Tensor, isleftT2rightD = True) -> torch.Tensor:
    # box1 的shape为(1, 4), box2的shape为(None, 4)
    # 防止输入错误
    box1 = box1.view(-1,4)
    box2 = box2.view(-1,4)
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
#       两组框的一一交并比                       
#==================================#
def iouOne2One(box1, box2, xyxy=False):
    box1 = box1.view(-1, 4)
    box2 = box2.view(-1, 4)
    if not xyxy:
        box1 = torch.concat([box1[:,:2] - box1[:,2:4] / 2, box1[:,:2] + box1[:,2:4] / 2], 1).to(CONST.device)
        box2 = torch.concat([box2[:,:2] - box2[:,2:4] / 2, box2[:,:2] + box2[:,2:4] / 2], 1).to(CONST.device)
    res = torch.zeros(box1.shape[0])
    for i in range(box1.shape[0]):
        res[i] = iou(box1[i], box2[i])
    return res
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
        ious = iou(box[sortIndex[0]], box[sortIndex[1:]])
        sortIndex = sortIndex[1:][ious < threshold]
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

# 中点长宽框转换为左上角右下角框
def xywh2xyxy(boxes):
    return torch.cat(((boxes[:,0:2] -  boxes[:,2:4] / 2), (boxes[:,0:2] +  boxes[:,2:4] / 2)), axis=1)
# 左上角右下角转换为框中点长宽框
def xyxy2xywh(boxes):
    return torch.cat((((boxes[:,0:2] +  boxes[:,2:4]) / 2), (boxes[:,2:4] - boxes[:,0:2])), axis=1)