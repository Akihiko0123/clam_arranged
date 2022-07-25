# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchsummary import summary
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 下のresnet50_baseline関数から呼び出される。nn.Moduleの子クラス
class ResNet_Baseline(nn.Module):

    # ブロックとlayer1,2,3におけるブロックの数が引数
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        # 2次元畳み込み1(入力=3,出力=64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # バッチ正規化1
        self.bn1 = nn.BatchNorm2d(64)
        # Relu関数
        self.relu = nn.ReLU(inplace=True)
        # maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 層１：下の_make_layer関数を読み込みボトルネックのブロックを、引数のリストの回数(3)だけ繰り返して作る。
        # blockは、上のBottleneck_baselineクラスで作った３層で真ん中が小さくなっている構造
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 層２：下の_make_layer関数を読み込みボトルネックのブロックを、引数のリストの回数(4)だけ繰り返して作る。
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 層３：下の_make_layer関数を読み込みボトルネックのブロックを、引数のリストの回数(6)だけ繰り返して作る。
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 平均pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        for m in self.modules():
            # modulesの中のmが畳み込みだった場合
            if isinstance(m, nn.Conv2d):
                # Heの初期値で重みを初期化する。(ReLuを用いる)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # modulesの中のmがバッチ正規化だった場合、平均重み1、バイアス0にする。
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # blocksの数字だけ繰り返し、ブロックを追加する。
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # 試しprint() x 3
        '''
        確認済：x.size(0) -> 512 or バッチの内の余り部分の場合 471等
        確認済：x.view(x.size(0),-1).size() -> torch.Size([512, 1024]) [512はバッチの内の余り部分の場合は471等になる]
        確認済：(例) x.view: tensor([[0.0778, 0.0108, 0.0118,  ..., 0.0207, 0.0183, 0.0522],
                                    [0.0636, 0.0269, 0.0245,  ..., 0.0154, 0.0113, 0.0537],
                                    [0.0978, 0.0143, 0.0211,  ..., 0.0170, 0.0143, 0.0226],
                                    ...,
                                    [0.0513, 0.0421, 0.0036,  ..., 0.0047, 0.0013, 0.0996],
                                    [0.0792, 0.0521, 0.0054,  ..., 0.0056, 0.0031, 0.1079],
                                    [0.0654, 0.0812, 0.0021,  ..., 0.0014, 0.0036, 0.0901]])
        '''
#        print("xsize(0):",x.size(0))
#        print("xsize(1):",x.view(x.size(0),-1).size())
#        print("x.view:",x.view(x.size(0),-1))
        x = x.view(x.size(0), -1)

        return x

# extract_features_fp.pyより呼び出し
# create_heatmaps.pyからも呼び出し
def resnet50_baseline(pretrained=False):
    # カスタマイズしたResNet-50モデルを構築する。
    # 引数がTrueなら、ImageNetで事前学習したモデルを返す。
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # ResNet_Baselineクラスを呼び出す。
    # Bottleneck_Baselineはブロックで、layerの中にBottleneck_Baselineクラスのボトルネック構造を入れ込んでいる。
    # [3,4,6,3]は各層におくブロックの数
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    # 事前学習されている場合は、事前学習された重みを読み込むために、下のload_pretrained_weights関数を呼び出し、modelに重みを加える。
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    # model_urlsの辞書から、指定した名前("resnet50"等)をキーに、URLを読み込んでmodelに重みを与える。
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model


