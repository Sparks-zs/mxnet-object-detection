import os, sys
import numpy as np
import xml.etree.ElementTree as ET
import mxnet
import argparse
import subprocess


class FruitDataset():
    def __init__(self, root_path, image_set, class_names='apple,banana,orange'):
        """
        :param root_path: 数据集所在根目录
        :param image_set: 数据属性 train or val
        :param class_names: 类别名称，默认全部导入
        """

        curr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 当前项目根目录
        assert os.path.exists(curr_path), 'Path does not exist: {}'.format(curr_path)  # 检查路径是否存在

        self.root_path = root_path
        self.data_path = os.path.join(curr_path, self.root_path, image_set)
        self.image_set = image_set
        self.class_names = class_names.strip().split(',')
        self.img_num = 0
        self.img_list = []
        self.img_shape = []
        self.xml_list = []
        self._load_img_xml_path()
        self.labels = self._load_image_labels()

    def _load_img_xml_path(self):
        """
         加载img和xml文件，生成分别包含image和xml文件的两个列表
        """
        data_list = os.listdir(self.data_path)
        # data_list.sort(key=lambda x: int(x[-6:-4]) if x[-6].isdigit() else int(x[-5]))  # 按图片序号从小到大排序

        img_list = []
        xml_list = []
        for idx, dir in enumerate(data_list):
            format = dir.split('.')[1]      # 文件格式
            if format == 'jpg':
                img_list.append(dir)
            elif format == 'xml':
                xml_list.append(dir)
        self.img_num = len(img_list)
        self.img_list = img_list
        self.xml_list = xml_list

    def _load_image_labels(self):
        """
        加载图片标签，存入self.labels变量中
        :return 返回图片标签
        """
        assert len(self.img_list) == len(self.xml_list), 'missing image or xml file'

        temp = []
        xml_nums = len(self.xml_list)

        for i in range(xml_nums):
            assert self.img_list[i].split('.')[0] == self.xml_list[i].split('.')[0], \
                'this {} does not match {}'.format(self.img_list[i], self.xml_list[i])   # img和xml文件是否一一对应

            label_file = os.path.join(self.data_path, self.xml_list[i])     # xml文件路径
            tree = ET.parse(label_file)              # 解析xml文件
            root = tree.getroot()                    # 获得第一标签
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            self.img_shape.append([width, height])
            label = []

            for obj in root.iter('object'):
                difficult = int(obj.find('difficult').text)
                # if not self.config['use_difficult'] and difficult == 1:
                #     continue
                cls_name = obj.find('name').text
                cls_id = self.class_names.index(cls_name)   # 查找当前class_name的序号
                xml_box = obj.find('bndbox')
                assert width != 0, 'error file is {}'.format(label_file)

                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax])
            temp.append(np.array(label))
        return temp

    def label_from_name(self, name):
        """
        :param index:   标签编号
        :return: 当前编号标签
        """
        assert self.labels is not None, "Labels not processed"
        name_idx = self.img_list.index(name)
        return name_idx, self.labels[name_idx]

    def image_path_from_name(self, name):
        """
        :param index:   图片编号
        :return: 当前编号图片路径
        """
        assert self.img_list is not None, 'Dataset not initialized '
        name_idx = self.img_list.index(name)
        name = self.img_list[name_idx]
        path = os.path.join(self.image_set, name)
        assert path, 'path does not exist {}'.format(path)
        return path

    def save_img_list(self, target_path, shuffle=True):
        """
        生成lst文件，保存指定路径中
        :param target_path: 目标路径
        :param train_ratio: 训练集占总数据集的比例
        :param shuffle: 是否打乱图片顺序
        """

        img_list = self.img_list.copy()     # 不加copy(),shuffle时一样会更改self.img_list变量
        if shuffle:
            import random
            random.shuffle(img_list)

        img_set_list = {str(self.image_set): img_list}

        # if 0.0 < train_ratio < 1.0:
        #     num = self.img_num * train_ratio
        #     img_set_list = {'train': img_list[:int(num)], 'val': img_list[int(num):]}
        # elif train_ratio == 0:
        #     img_set_list = {'val': img_list}

        for img_set in img_set_list:
            str_list = []
            for idx, name in enumerate(img_set_list[img_set]):
                i, label = self.label_from_name(name)      # 图片标签
                img_shape = self.img_shape[i]
                path = self.image_path_from_name(name)  # 图片路径
                str_list.append('\t'.join([str(idx), str(4),
                                           str(label.shape[1]), str(img_shape[0]), str(img_shape[1])] +
                    ['{0:.4f}'.format(x) for x in label.ravel()] + [path]) + '\n')

            if str_list:
                fname = os.path.join(target_path, img_set + '.lst')
                with open(fname, 'w+') as f:        # 写入.lst文件中
                    for line in str_list:
                        f.write(line)
            else:
                raise RuntimeError("No image in this file")

# example:
#
# root_path = 'data/Fruit'
# target_path = 'D:/mxnet_projects/mxnet_ssd/data/Fruit'
# fruitdataset = FruitDataset(root_path, 'train')
# fruitdataset.save_img_list(target_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lst and rec for dataset')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=None, type=str)
    parser.add_argument('--target', dest='target_path', help='output list path',
                        default=None, type=str)
    parser.add_argument('--set', dest='set', help='train or test',
                        default='train', type=str)
    # parser.add_argument('--train-ratio', dest='train_ratio', help='train/val',
    #                     default=1.0, type=float)
    parser.add_argument('--class-names', dest='class_names', help='choice class to use',
                        default='apple,banana,orange', type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        default=True, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("saving list to disk...")
    fruitdataset = FruitDataset(args.root_path, args.set)
    fruitdataset.save_img_list(args.target_path, args.shuffle)
    print('{} list file {} is generated ...'.format(args.set, args.target_path))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(['python', im2rec_path,
                           os.path.abspath(args.target_path),
                           os.path.abspath(args.root_path)])
    print('Record file  is generated ...')

    # example:
    # python tools/fruit_dataset.py --root data/Fruit --target data/Fruit
    # root_path = 'D:/mxnet_projects/mxnet_ssd/data/Fruit'
    # target_path = 'D:/mxnet_projects/mxnet_ssd/data/Fruit/train.lst'