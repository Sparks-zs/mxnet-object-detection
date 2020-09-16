import mxnet as mx
from mxnet.gluon import nn


class TinyYolov1(nn.HybridBlock):
    def __init__(self, num_classes):
        super(TinyYolov1, self).__init__()
        self.num_classes = num_classes

        with self.name_scope():
            self.conv_layers = nn.HybridSequential()
            with self.name_scope():
                self.conv_layers = nn.HybridSequential()
                self.conv_layers.add(nn.Conv2D(16, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(32, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(128, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(256, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(512, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(1024, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),

                                     nn.Conv2D(256, kernel_size=3, strides=1, padding=1),
                                     nn.BatchNorm(),
                                     nn.LeakyReLU(0.1),
                                     nn.MaxPool2D(pool_size=2, strides=2),
                                     )

            self.conn_layers = nn.HybridSequential()
            self.conv_layers.add(nn.Dense(1470),
                                 nn.Conv2D((self.num_classes + 2 * 10) * 7 * 7, kernel_size=1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv_layers(x)
        x = self.conn_layers(x)
        return x



x = mx.nd.zeros(shape=(320,3,448,448))
net = Yolov1(1)
net.initialize()
out = net(x)
print(out.shape)