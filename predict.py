import os
import argparse
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import image, nd
from tools.tools import try_gpu, import_module


# 读取单张测试图片
def single_image_data_loader(filename, test_image_size=300):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """

    def reader():
        img_size = test_image_size
        file_path = os.path.join(filename)
        img = image.imread(file_path)
        img = image.imresize(img, img_size, img_size, 3).astype('float32')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = nd.array(mean).reshape((1, 1, -1))
        std = nd.array(std).reshape((1, 1, -1))
        out_img = (img / 255.0 - mean) / std
        out_img = out_img.transpose((2, 0, 1)).expand_dims(axis=0)    # 通道 h w c->c h w

        yield out_img
    return reader


# 预测目标
def predict(test_image, net, img, labels, threshold=0.3):
    anchors,bbox_preds,cls_preds= net(test_image)
    cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0, 2, 1)), mode='channel')
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors,
                                          force_suppress=True, clip=True,
                                          threshold=0.5, nms_threshold=.45)

    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx:
        output = output[0, idx]
        display(img, labels, output, threshold=threshold)
        return True
    else:
        return False


# 显示多个边界框
def show_bboxes(axes, bboxes, labels=None):
    for i, bbox in enumerate(bboxes, 0):
        bbox = bbox.asnumpy()
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, linewidth=2, color='w')
        axes.add_patch(rect)
        if labels:
            axes.text(rect.xy[0], rect.xy[1], labels,
                      horizontalalignment='center', verticalalignment='center', fontsize=8,
                      color='k', bbox=dict(facecolor='w', alpha=1))


def display(img, labels, output, threshold):
    fig = plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        label = labels[int(row[0].asscalar())]
        show_bboxes(fig.axes, bbox, '%s-%.2f' % (label, score))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='predict the single image')
    parser.add_argument('--image-path', dest='img_path', help='image path',
                        default=None, type=str)
    parser.add_argument('--model', dest='model', help='choice model to use',
                        default='resnet_ssd', type=str)
    parser.add_argument('--model-params', dest='model_params', help='choice model params to use',
                        default='mask_resnet18_SSD_model.params', type=str)
    parser.add_argument('--class-names', dest='class_names', help='choice class to use',
                        default='without_mask,with_mask,mask_weared_incorrect', type=str)
    parser.add_argument('--image-shape', dest='image_shape', help='image shape',
                        default=512, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # ctx = try_gpu()
    ctx = mx.cpu()

    img = image.imread(args.img_path).as_in_context(ctx)
    reader = single_image_data_loader(args.img_path, args.image_shape)
    labels = args.class_names.strip().split(',')
    class_nums = len(labels)

    model_path = os.path.join('model', args.model_params)
    net = import_module('model.'+args.model).get_model(class_nums, pretrained_model=model_path, pretrained=True, ctx=ctx)

    for x in reader():
        output = predict(x, net, img, labels)
        if not output:
            print('not found!')


# img_path = 'D:/mxnet_projects/mxnet_ssd/data/facemask/images/maksssksksss5.png'