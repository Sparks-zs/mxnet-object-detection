from mxnet import contrib, nd, init
from mxnet.gluon import nn

# 类别预测层
# 输出通道数=以某一坐标为中心的所有锚框的类别预测
# num_classes+1 预测类别+背景
def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes+1),
                     kernel_size=3, padding=1)


# 边界框预测层
# 输出通道数=所有锚框的4个偏移量
def bbox_predictor(num_achors):
    return nn.Conv2D(num_achors * 4,
                     kernel_size=3, padding=1)


# 连接多尺度的预测
# 由于每个尺度上特征图的形状或以同一单元为中心生成的锚框个数都可能不同，
# 因此不同尺度的预测输出形状可能不同。

# 以(批量大小, 宽×高×通道数)的统一格式转换二维,方便后续连接
def flatten_pred(pred):
    return pred.transpose((0, 2, 3, 1)).flatten()


# 连接column轴
def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


# 高和宽减半块
# 使输出特征图中每个单元的感受野更宽广
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                # nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk


# 基础网络块
# 串联3个高和宽减半块
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk
# def base_net():
#     from mxnet.gluon.model_zoo.vision import vgg16
#     vgg16net = vgg16(pretrained=True)
#     net = nn.HybridSequential()
#     for layer in vgg16net.features[:-4]:
#         net.add(layer)
#     return net

# 完整的模型

# 基础网络块--高和宽减半块*3--全局最大池化
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


# 各个模块前向计算的顺序
# 返回特征图Y, 锚框, 类别预测, 偏移量
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio,)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)

    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# 定义SSD模型
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes

        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self,'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        return (nd.concat(*anchors, dim=1),
                concat_preds(bbox_preds),
                concat_preds(cls_preds).reshape(0, -1, self.num_classes+1))

# net = TinySSD(1)
# net.initialize(init=init.Xavier())
# x = nd.uniform(shape=(32,3,300,300))
# net(x)
# print(net)
#
# for k, v in net.collect_params().items():
#     print(v)
#     print(v.data())