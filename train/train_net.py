import mxnet as mx
from mxnet import autograd, contrib, gluon, nd
from train.smooth_l1 import smooth_l1, FocalLoss
from utils import utils
from tools.draw_details import draw_details
from model.vgg_ssd import get_model
from model.resnet_ssd import get_model

import os
import time
import logging


# 主训练函数
def train(data_path, net, num_classes, data_size, batch_size, epochs, wd, momentum, lr, save_model_path, log_file_path, ctx=mx.cpu()):

    train_rec_path = os.path.join(data_path, 'train.rec')
    train_idx_path = os.path.join(data_path, 'train.idx')
    val_rec_path = os.path.join(data_path, 'val.rec')

    augs = mx.image.CreateDetAugmenter(data_shape=(3, data_size, data_size),
                                       rand_crop=1,
                                       min_object_covered=0.9,
                                       aspect_ratio_range=(0.5, 2),
                                       area_range=(0.1, 1.5),
                                       max_attempts=100,
                                       rand_mirror=True,
                                       rand_gray=0.2,
                                       brightness=0.5,
                                       contrast=0.5,
                                       saturation=0.5,
                                       rand_pad=0.4,
                                       hue=0.5,
                                       mean=True,
                                       std=True,
                                       )

    train_iter = mx.image.ImageDetIter(
        path_imgidx=train_idx_path,
        path_imgrec=train_rec_path,
        batch_size=batch_size,
        data_shape=(3, data_size, data_size),
        shuffle=True,
        aug_list=augs,
    )

    val_iter = mx.image.ImageDetIter(
        path_imgrec=val_rec_path,
        batch_size=batch_size,
        data_shape=(3, data_size, data_size),
        shuffle=False,
        mean=True,
        std=True
    )

    # net = get_ssd_model(num_classes, pretrained_base=True)
    # net = get_model(num_classes, pretrained_base=True, ctx=ctx)
    net.collect_params().reset_ctx(ctx=ctx)
    net.hybridize()

    # lrs = mx.lr_scheduler.FactorScheduler(step=200, factor=0.8, stop_factor_lr=lr, base_lr=lr)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd, 'momentum': momentum})

    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # cls_loss = FocalLoss()
    bbox_loss = smooth_l1()

    def evaluate_accuracy(data_iter, net, ctx):
        """
        :param data_iter: 数据集加载器
        :param net: 模型网络
        :param ctx: 可使用的gpu列表
        :return: 验证集准确率
        """

        data_iter.reset()
        outs, labels = None, None
        for batch in data_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)

            anchors,bbox_preds,cls_preds = net(X)
            # 为每个锚框标注类别和偏移量
            cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0, 2, 1)), mode='channel')
            out = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors,
                                               force_suppress=True, clip=False, nms_threshold=0.45)
            if outs is None:
                outs = out
                labels = Y
            else:
                outs = nd.concat(outs, out, dim=0)
                labels = nd.concat(labels, Y, dim=0)

            AP = utils.evaluate_MAP(outs, labels)

            return AP

    # set up logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode='w')
    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    for epoch in range(epochs):

        ce_metric.reset()
        smoothl1_metric.reset()

        # if epoch == 100 or epoch == 150:
        #     trainer.set_learning_rate(trainer.learning_rate * 0.1)
        #     print('the new lr is:', trainer.learning_rate)
        #     net.save_parameters('D:/mxnet_projects/mxnet_ssd/model/{}_bottle_SSD_model.params'.format(epoch))

        train_iter.reset()  # 从头读取数据
        btic = time.time()

        for i, batch in enumerate(train_iter):
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, bbox_preds, cls_preds = net(X)
                # 为每个锚框标注类别和偏移量
                bbox_labels, bbox_masks, cls_labels = contrib.nd.MultiBoxTarget(
                    anchors, Y, cls_preds.transpose((0, 2, 1)),
                    negative_mining_ratio=3, negative_mining_thresh=.5)
                # 根据类别和偏移量的预测和标注值计算损失函数
                cls = cls_loss(cls_preds, cls_labels)
                bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
                l = cls + bbox
            l.backward()
            trainer.step(batch_size)

            if i % 50 == 0:
                ce_metric.update(0, cls)
                smoothl1_metric.update(0, bbox)
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                val_AP = evaluate_accuracy(val_iter, net, ctx)
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.2e}, {}={:.2e}, val_AP={:.3f}'.format(
                    epoch, i, batch_size / (time.time() - btic), name1, loss1, name2, loss2, val_AP))
            btic = time.time()

    net.save_parameters(save_model_path)


# train('D:/mxnet_projects/mxnet_ssd/data/facemask',
#       3, 512, 8, 60, 5e-4, 0.9, 0.01,
#       'D:/mxnet_projects/mxnet_ssd/model/mask_resnet18_SSD_model.params', 'D:/mxnet_projects/mxnet_ssd/log/resnet18_SSD_train.log', mx.gpu())