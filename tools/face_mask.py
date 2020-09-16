import os, sys
import numpy as np
import xml.etree.ElementTree as ET
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from tools.imdb import imdb
import mxnet
import argparse
import subprocess



class FaceMask(imdb):
    def __init__(self, mask_path, image_set, class_names, shuffle):
        super(FaceMask, self).__init__()

        self.mask_path = mask_path
        self.image_set = image_set
        self.class_names = class_names.strip().split(',')  # 类别列表
        self.num_class = len(self.class_names)
        self.image_shape_labels = []
        self.image_index = self._load_image_index(shuffle)  # 索引列表
        self.num_images = len(self.image_index)
        self.labels = self._load_image_labels()  # 标签
        print(self.class_names)


    def _load_image_index(self, shuffle):
        image_index = []
        image_path = os.path.join(self.mask_path, self.image_set, 'images')
        image_file = os.listdir(image_path)
        for name in image_file:
            idx = name.split('maksssksksss')[1][:-4]
            image_index.append(idx)

        if shuffle:
            import random
            random.shuffle(image_index)
        return image_index

    def image_path_from_index(self, index):
        image_file = os.path.join(self.image_set, 'images', 'maksssksksss' + str(self.image_index[index]) + '.png')
        assert image_file, 'path {} is not exist'.format(image_file)
        return image_file

    def label_from_index(self, index):
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def image_shape_from_index(self, index):
        assert self.image_shape_labels is not None, "Image shape labels not processed"
        return self.image_shape_labels[index]

    def _label_path_from_index(self, index):
        label_file = os.path.join(self.mask_path, 'annotations', 'maksssksksss' + str(index) + '.xml')
        assert label_file, 'path {} is not exist'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        加载图片标签，存入self.image_labels变量中
        :return 返回图片标签
        """
        temp = []

        for idx in self.image_index:
            label_file = self._label_path_from_index(idx)  # 返回该图片的annotation文件路径
            tree = ET.parse(label_file)  # 解析xml文件
            root = tree.getroot()  # 获得第一标签
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            self.image_shape_labels.append([width, height])
            label = []

            for obj in root.iter('object'):
                # difficult = int(obj.find('difficult').text)
                # if not self.config['use_difficult'] and difficult == 1:
                #     continue
                cls_name = obj.find('name').text
                if cls_name not in self.class_names:
                    continue
                    # self.class_names.append(cls_name)
                cls_id = self.class_names.index(cls_name)  # 查找当前class_name的序号
                xml_box = obj.find('bndbox')

                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax])
            temp.append(np.array(label))
        return temp


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lst and rec for dataset')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=None, type=str)
    parser.add_argument('--target', dest='target_path', help='output list path',
                        default=None, type=str)
    parser.add_argument('--set', dest='set', help='train, val',
                        default='train,val', type=str)
    parser.add_argument('--class-names', dest='class_names', help='choice class to use',
                        default='without_mask,with_mask,mask_weared_incorrect', type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        default=False, type=bool)
    args = parser.parse_args()
    return args


def load_facemask(mask_path, target_path, image_set, class_names, shuffle):
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"

    for s in image_set:
        imdb = FaceMask(mask_path, s, class_names, shuffle)
        imdb.save_img_list(target_path, shuffle)


if __name__ == '__main__':
    args = parse_args()
    print("saving list to disk...")
    load_facemask(args.root_path, args.target_path, args.set, args.class_names, args.shuffle)
    print('{} list file {} is generated ...'.format(args.set, args.target_path))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(['python', im2rec_path,
                           os.path.abspath(args.target_path),
                           os.path.abspath(args.root_path),
                           '--pack-label'])
    print('Record file  is generated ...')

    # python tools/face_mask.py --root data/facemask --target data/facemask
# data_path = 'D:/mxnet_projects/mxnet_ssd/data/facemask'
# image_path = os.path.join(data_path, 'images')
# image_num = len(os.listdir(image_path))
# image_index = list(range(0, image_num))
# import random
# random.shuffle(image_index)
#
# num = int(image_num*0.9)
# train_list = image_index[:num]
# val_list = image_index[num:]
#
# data_list = {'train':train_list, 'val':val_list}
# import shutil
#
# for set_list in data_list:
#     set_path = os.path.join(data_path, set_list)
#     if not os.path.exists(set_path):
#         os.mkdir(set_path)
#
#     for idx in data_list[set_list]:
#         xml_dir = os.path.join(set_path, 'annotations')
#         if not os.path.exists(xml_dir):
#             os.mkdir(xml_dir)
#         img_dir = os.path.join(set_path, 'images')
#         if not os.path.exists(img_dir):
#             os.mkdir(img_dir)
#
#         xml_name = 'maksssksksss' + str(idx) + '.xml'
#         xml_path = os.path.join(data_path, 'annotations', xml_name)
#         img_name = 'maksssksksss' + str(idx) + '.png'
#         img_path = os.path.join(data_path, 'images', img_name)
#
#         shutil.copy(xml_path, os.path.join(xml_dir, xml_name))
#         shutil.copy(img_path, os.path.join(img_dir, img_name))


