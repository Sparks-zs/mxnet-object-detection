import mxnet as mx
from mxnet import nd, init
from mxnet.gluon import nn

vgg_spec = {
    16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
}

extra_spec = {
    300: [((256, 1, 1, 0), (512, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 1, 0)),
          ((128, 1, 1, 0), (256, 3, 1, 0))]
}

layers, filters = vgg_spec[16]
extras = extra_spec[300]


class Normalize(nn.HybridBlock):
    """Normalize layer described in https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    n_channel : int
        Number of channels of input.
    initial : float
        Initial value for the rescaling factor.
    eps : float
        Small value to avoid division by zero.

    """
    def __init__(self, n_channel, initial=1, eps=1e-5):
        super(Normalize, self).__init__()
        self.eps = eps
        with self.name_scope():
            self.scale = self.params.get('normalize_scale', shape=(1, n_channel, 1, 1),
                                         init=mx.init.Constant(initial))

    def hybrid_forward(self, F, x, scale):
        x = F.L2Normalization(x, mode='channel', eps=self.eps)
        return F.broadcast_mul(x, scale)


class VGG_atrous(nn.HybridBlock):
    def __init__(self):
        super(VGG_atrous, self).__init__()

        self.init = {
            'weight_initializer': init.Xavier(
                rnd_type='gaussian', factor_type='out', magnitude=2),
            'bias_initializer': 'zeros'
        }
        with self.name_scope():
            init_scale = mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) * 255
            self.init_scale = self.params.get_constant('init_scale', init_scale)
            self.stages = nn.HybridSequential()
            for l, f in zip(layers, filters):
                stage = nn.HybridSequential(prefix='')
                with stage.name_scope():
                    for _ in range(l):
                        stage.add(nn.Conv2D(f, kernel_size=3, padding=1, **self.init))
                        stage.add(nn.Activation('relu'))
                self.stages.add(stage)

            stage = nn.HybridSequential(prefix='dilated_')
            with stage.name_scope():
                stage.add(nn.Conv2D(1024, kernel_size=3, padding=6, dilation=6, **self.init))
                stage.add(nn.Activation('relu'))
                stage.add(nn.Conv2D(1024, kernel_size=1, **self.init))
                stage.add(nn.Activation('relu'))

            self.stages.add(stage)
            self.norm4 = Normalize(filters[3], 20)

            self.extras = nn.HybridSequential()
            for i, config in enumerate(extras):
                extra = nn.HybridSequential(prefix='extra%d_'%(i))
                with extra.name_scope():
                    for f, k, s, p in config:
                        extra.add(nn.Conv2D(f, k, s, p, **self.init))
                        extra.add(nn.Activation('relu'))
                self.extras.add(extra)

    def hybrid_forward(self, F, x, init_scale):
        x = F.broadcast_mul(x, init_scale)
        assert len(self.stages) == 6
        outputs = []
        for stage in self.stages[:3]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        x = self.stages[3](x)
        norm = self.norm4(x)
        outputs.append(norm)
        x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                      pooling_convention='full')
        x = self.stages[4](x)
        x = F.Pooling(x, pool_type='max', kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      pooling_convention='full')
        x = self.stages[5](x)
        outputs.append(x)
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        return outputs


class SSD(nn.HybridBlock):
    def __init__(self, num_classes):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                       [1, 2, .5], [1, 2, .5]]

        self.features = VGG_atrous()

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
        outputs = self.features(x)
        anchors, cls_preds, bbox_preds = [None] * 6, [None] * 6, [None] * 6
        for i, x in enumerate(outputs):
            cls_preds[i] = self.cls_predictor[i](x)
            bbox_preds[i] = self.bbox_predictor[i](x)
            anchors[i] = F.contrib.MultiBoxPrior(x, sizes=self.sizes[i], ratios=self.ratios[i])

        bbox_preds = self.concat_preds(F, bbox_preds)
        cls_preds = self.concat_preds(F, cls_preds).reshape((0, -1, self.num_classes + 1))
        anchors = F.concat(*anchors, dim=1)

        return anchors, bbox_preds, cls_preds


def get_model(num_classes, pretrained_model=None, pretrained=False, pretrained_base=False, ctx=mx.gpu()):
    net = SSD(num_classes)
    if pretrained_base:
        net.initialize(init=init.Xavier(), ctx=ctx)
        pretrained_base_model = 'D:/mxnet_projects/mxnet_ssd/model/vgg16_atrous_300.params'
        net.features.load_parameters(pretrained_base_model, allow_missing=True)
    elif pretrained:
        net.load_parameters(pretrained_model, ctx=ctx)
    return net