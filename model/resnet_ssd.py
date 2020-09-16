import mxnet as mx
from mxnet.gluon import nn

resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512])}
# features=['stage3_activation1', 'stage4_activation1']
num_filters = [512, 512, 256, 256]


class BasicBlockV1(nn.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0):
        super(BasicBlockV1, self).__init__()

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                                use_bias=False, in_channels=in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1,
                                use_bias=False, in_channels=channels))
        self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(residual+x, act_type='relu')

        return x


class ResNetV1(nn.HybridBlock):
    def __init__(self, block, layers, channels, classes=1000):
        super(ResNetV1, self).__init__()
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels[0], kernel_size=7, strides=2, padding=3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1], stride,
                                                 i+1, in_channels=channels[i]))

            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


class resnet_SSD(nn.HybridBlock):
    def __init__(self, num_classes, resnet):
        super(resnet_SSD, self).__init__()

        self.num_classes = num_classes
        self.sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                       [1, 2, .5], [1, 2, .5]]

        self.features = resnet.features

        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        self.extras = nn.HybridSequential(prefix='extras')
        for i, f in enumerate(num_filters):
            extra = nn.HybridSequential(prefix='extra%d_' % i)
            with extra.name_scope():
                extra.add(nn.Conv2D(f, kernel_size=1, strides=1, weight_initializer=weight_init))
                extra.add(nn.Activation('relu'))
                extra.add(nn.Conv2D(f, kernel_size=3, strides=2, padding=1, weight_initializer=weight_init))
                extra.add(nn.Activation('relu'))
            self.extras.add(extra)

        self.bbox_predictor = nn.HybridSequential()
        self.cls_predictor = nn.HybridSequential()

        for s, r in zip(self.sizes, self.ratios):
            num_anchors = len(s) + len(r) - 1  # 生成的锚框数量
            self.bbox_predictor.add(nn.Conv2D(num_anchors * 4,
                                              kernel_size=3, padding=1))
            self.cls_predictor.add(nn.Conv2D(num_anchors * (self.num_classes + 1),
                                             kernel_size=3, padding=1))

    # 以(批量大小, 宽×高×通道数)的统一格式转换二维,方便后续连接
    def flatten_pred(self, pred):
        return pred.transpose((0, 2, 3, 1)).flatten()

    # 连接column轴
    def concat_preds(self, F, preds):
        return F.concat(*[self.flatten_pred(p) for p in preds], dim=1)

    def hybrid_forward(self, F, x):
        outputs = []
        for feature in self.features[:7]:
            x = feature(x)
        outputs.append(x)
        x = self.features[7](x)
        outputs.append(x)
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        anchors, cls_preds, bbox_preds = [None] * 6, [None] * 6, [None] * 6
        for i, x in enumerate(outputs):
            cls_preds[i] = self.cls_predictor[i](x)
            bbox_preds[i] = self.bbox_predictor[i](x)
            anchors[i] = F.contrib.MultiBoxPrior(x, sizes=self.sizes[i], ratios=self.ratios[i])

        bbox_preds = self.concat_preds(F, bbox_preds)
        cls_preds = self.concat_preds(F, cls_preds).reshape((0, -1, self.num_classes + 1))
        anchors = F.concat(*anchors, dim=1)

        return anchors, bbox_preds, cls_preds


def get_resnet(num_layers):
    block_type, layers, channels = resnet_spec[num_layers]
    net = ResNetV1(BasicBlockV1, layers, channels)
    return net


def get_model(num_classes, num_layer=18, pretrained_model=None, pretrained=False, pretrained_base=False, pretrianed_base_path=None, ctx=mx.gpu()):
    resnet = get_resnet(num_layer)
    net = resnet_SSD(num_classes, resnet)
    if pretrained_base:
        net.initialize(init=mx.init.Xavier(), ctx=ctx)
        assert pretrianed_base_path, '预训练模型路径不能为空'
        resnet.load_parameters(pretrianed_base_path)
    elif pretrained:
        net.load_parameters(pretrained_model, ctx=ctx)
    return net


# net = get_resnet_ssd_model(1, pretrained_base=True, ctx=mx.cpu())
# x = mx.nd.uniform(shape=(32,3,512,512))
# anchors, bbox_preds, cls_preds = net(x)
#
# print('anchors shape:', anchors.shape)
# print('bbox_preds shape:', bbox_preds.shape)
# print('cls_preds shape:', cls_preds.shape)
# print(net.features[0].weight.data())