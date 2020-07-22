# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:49:24 2018

@author: 60236
"""
import torch.nn as nn

from .backbone import ResNet50
from .layers import  conv_bn, conv_bn_relu, conv_relu
from .corner_pooling import top_pool, left_pool,bottom_pool, right_pool

        
class corner_net(nn.Module):
    def __init__(self, num_classes, inplanes=256, backbone=ResNet50):
        super(corner_net,self).__init__()
        self.features = backbone()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes    

        self.conv_bn_relu1 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu2 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu3 = conv_bn_relu(inplanes,inplanes)
        
        self.conv_bn_relu4 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu5 = conv_bn_relu(inplanes,inplanes)
        self.conv_bn_relu6 = conv_bn_relu(inplanes,inplanes)
        
        self.conv_bn_tl = conv_bn(inplanes,inplanes)
        self.conv_bn_br = conv_bn(inplanes,inplanes)
        
        self.conv_bn_1x1_tl = conv_bn(inplanes,inplanes,1,1,0)
        self.conv_bn_1x1_br = conv_bn(inplanes,inplanes,1,1,0)
        
        self.conv_relu1 = conv_relu(inplanes,inplanes)
        self.conv_relu2 = conv_relu(inplanes,inplanes)
        self.conv_relu3 = conv_relu(inplanes,inplanes)
        self.conv_relu4 = conv_relu(inplanes,inplanes)
        self.conv_relu5 = conv_relu(inplanes,inplanes)
        self.conv_relu6 = conv_relu(inplanes,inplanes)
        
        self.out_ht_tl = nn.Conv2d(inplanes, num_classes,1,1,0)
        self.out_ht_br = nn.Conv2d(inplanes, num_classes,1,1,0)
        
        self.out_eb_tl = nn.Conv2d(inplanes,1,1,1,0)
        self.out_eb_br = nn.Conv2d(inplanes,1,1,1,0)
        
        self.out_of_tl = nn.Conv2d(inplanes,2,1,1,0)
        self.out_of_br = nn.Conv2d(inplanes,2,1,1,0)
        
    def forward(self,x):
        x = self.features(x)
        
        """
        top-left:
        首先从feature map中得到top pooling,left pooling以及feature map卷积后的表征，
        根据这个表征得到heatmap, embedding,offset三个值

        1. conv + top pooling
        2. conv + left pooling
        3. 将top pooling和left pooling相加经过conv
        4. Conv_BN_Relu(Relu(conv + top pooling和left pooling的表征))
        """
        a = self.conv_bn_relu1(x)
        a = top_pool()(a)
        b = self.conv_bn_relu2(x)
        b = left_pool()(b)
        ab = self.conv_bn_tl(a+b)

        c = self.conv_bn_1x1_tl(x)
        out = self.conv_bn_relu3(self.relu(c+ab))
        
        """
        首先，预测模型会输出一个热力图，大小为 H*W*C ，其中C代表类别数。
        热力图本质上就是一个二进制的mask，通道c上的热力图代表属于c类的物体的corner的位置。
        在训练过程中，只有位于ground truth boxes的corner才是正样本，其他的都是负样本，
        而在计算loss的时候，会给不同负样本分配不同的权重。
        这是因为，离正样本corner较近的那些corner构成的box也能够框出一个物体，
        如下图中的绿色虚线对应的左上角corner和右下角corner，这些corner也能框出物体，
        但是这些corner被当作负样本，那我们就减小这部分corner的penalty权重。
        """
        heatmaps_tl   = self.out_ht_tl(self.conv_relu1(out))

        """
        此外，网络还会给每个corner预测一个embedding vector，之后会根据这些embedding之间的距离来给corner分组。
        这是受associative embedding这篇文章的启发，即如果一个top-left corner和一个bottom-right corner属于一个box，
        那它们的embedding之间的距离应该很小。
        """
        embeddings_tl = self.out_eb_tl(self.conv_relu2(out))

        """
        预测offset是来对预测的corner的位置进行微调的。首先原图中的(x, y)点在经过CNN网络之后，
        会映射到热力图上的( x/n 的下取整， y/n 的下取整), n是下采样率，
        然后之后将热力图中预测的corner映射回原图时，就会因为这种取整操作导致精度上的偏差，
        这种偏差对于一些小的bounding box会产生比较大的影响。所以网络需要预测一个offset来调整位置
        """
        offsets_tl    = self.out_of_tl(self.conv_relu3(out))
        
        ##bottem-right
        i = self.conv_bn_relu4(x)
        i = bottom_pool()(i) 
        j = self.conv_bn_relu5(x)
        j = right_pool()(j)
        ij = self.conv_bn_br(i+j)
        k = self.conv_bn_1x1_br(x)
        out = self.conv_bn_relu6(self.relu(k+ij))
        
        heatmaps_br   = self.out_ht_br(self.conv_relu4(out))
        embeddings_br = self.out_eb_br(self.conv_relu5(out))
        offsets_br    = self.out_of_br(self.conv_relu6(out))

        return [heatmaps_tl,heatmaps_br,embeddings_tl,embeddings_br,offsets_tl,offsets_br]

    