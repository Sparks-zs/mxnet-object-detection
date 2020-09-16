import argparse
from train import train_net
from tools.tools import try_gpu, import_module


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lst and rec for dataset')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=None, type=str)
    parser.add_argument('--net', dest='net', help='choice net',
                        default='vgg_ssd', type=str)
    parser.add_argument('--pretrained-base-path', dest='pretrained_base_path', help='choice model params to use',
                        default='model/resnet18_v1_512.params', type=str)
    parser.add_argument('--data', dest='data_path', help='data path',
                        default=None, type=str)
    parser.add_argument('--num-classes', dest='num_classes', help='num class',
                        default=None, type=str)
    parser.add_argument('--data_size', dest='data_size', help='image data shape',
                        default=300, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',
                        default=8, type=int)
    parser.add_argument('--wd', dest='wd', help='wd',
                        default=5e-4, type=float)
    parser.add_argument('--momentum', dest='momentum', help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--save_model_path', dest='save_model_path', help='save model path',
                        default='model/vgg_ssd_model.params', type=str)
    parser.add_argument('--log_file_path', dest='log_file_path', help='log file path',
                        default='log/train.log', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    ctx = try_gpu()
    net = import_module('model.'+args.net).get_model(args.num_classe, pretrained_base=True,
                                                     pretrained_base_path=args.pretrained_base_path, ctx=ctx)
    train_net.train(args.data_path, net, args.num_classes, args.data_size, args.batch_size, args.epochs,
                    args.wd, args.momentum, args.lr, args.save_model_path, args.log_file_path, ctx)
