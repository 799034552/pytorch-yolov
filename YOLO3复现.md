# YOLO3复现

### 1. 先实现整体网络

YOLO3的网络大概是这样的：

![img](YOLO3复现.assets/2019040211084050.jpg)

#### 1.1 darknet53网络实现

darknet53网络长这样，包含了较多的残差块



![Ait](YOLO3复现.assets/20200314120801677.png)



1. 定义一个简单卷积层
2. 定义一个残差块
3. 定义darknet53类包含他们

```python
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
```

#### 1.2 conv Set的实现

看结构就是一堆卷积层而已，直接上代码

```python
#==================================#
#           convSet                    
#==================================#
class convSet(nn.Module):
    def __init__(self, inputC) -> None:
        super(convSet, self).__init__()
        tempC = inputC // 2
        self.m = nn.Sequential(
            self.conv(input, tempC, 1),
            self.conv(tempC, input, 3),
            self.conv(input, tempC, 1),
            self.conv(tempC, input, 3),
            self.conv(input, tempC, 1),
        )

    def conv(self, inputC, outputC, keralSize):
        return nn.Sequential(
            nn.Conv2d(inputC, outputC, keralSize, padding="same"),
            nn.BatchNorm2d(outputC),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.m(x)

```

#### 1.3最终输出层

就一个3*3的卷积加一个输出

```python
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
```



#### 迁移学习

```python
```

