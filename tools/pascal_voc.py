import os
import numpy as np
import xml.etree.ElementTree as ET
from tools.imdb import imdb


class PascalVoc(imdb):
    def __init__(self, devkit_path, image_set, year, class_names, shuffle):
        super(PascalVoc, self).__init__()
        """
        生成.rec文件需要的.lst文件
        :param devkit_path: VOCdevkit路径
        :param image_set:   数据集属性 'trainval train val test'
        :param year:        2007 2012
        :param class_names:     图片类别
        :param shuffle:     是否打乱顺序
        """
        self.devkit_path = devkit_path          # devkit数据集路径
        self.data_path = os.path.join(devkit_path, 'VOC'+year)
        self.image_set = image_set              # 数据集属性，train、val or test'
        self.year = year                        # 年份
        self.class_names = class_names.strip().split(',')          # 类别列表

        self.image_shape_labels = []                            # 图片宽和高的列表
        self.image_index = self._load_image_index(shuffle)      # 索引列表
        self.num_images = len(self.image_index)                 # 图片数量
        self.labels = self._load_image_labels()                 # 标签

    def _load_image_index(self, shuffle):
        """
        加载图片索引
        :param shuffle: 是否打乱索引
        :return 返回索引列表
        """
        image_index = []
        for cls_name in self.class_names:
            image_index_file = os.path.join(self.data_path, 'ImageSets', 'Main',
                                            cls_name+'_'+self.image_set+'.txt')
            assert os.path.exists(image_index_file), 'path {} is not exist'.format(image_index_file)

            with open(image_index_file) as f:
                for x in f.readlines():
                    if int(x[-3:]) == -1:
                        continue
                    x = x.split(' ')[0]
                    image_index.append(x)
                    # if len(image_index) == 10:
                    #     break

        if shuffle:
            import random
            random.shuffle(image_index)
        return image_index

    def label_from_index(self, index):
        """
        :param index:   标签编号
        :return: 当前编号标签
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def image_shape_from_index(self, index):
        """
        :param index:   标签编号
        :return: 当前编号标签
        """
        assert self.image_shape_labels is not None, "Image shape labels not processed"
        return self.image_shape_labels[index]

    def image_path_from_index(self, index):
        """
        返回图片路径
        :param index: 图片索引
        :return: 当前索引的图片路径
        """
        image_file = os.path.join('VOC'+self.year, 'JPEGImages', self.image_index[index]+'.jpg')
        assert image_file, 'path {} is not exist'.format(image_file)
        return image_file

    def _label_path_from_index(self, index):
        """
        返回标注文件路径
        :param index: 图片索引
        :return: 当前索引的标注文件路径
        """
        label_file = os.path.join(self.data_path, 'Annotations', str(index)+'.xml')
        assert label_file, 'path {} is not exist'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        加载图片标签，存入self.image_labels变量中
        :return 返回图片标签
        """
        temp = []

        for idx in self.image_index:
            label_file = self._label_path_from_index(idx)   # 返回该图片的annotation文件路径
            tree = ET.parse(label_file)             # 解析xml文件
            root = tree.getroot()                   # 获得第一标签
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
                    cls_id = 1

                cls_id = self.class_names.index(cls_name)  # 查找当前class_name的序号
                xml_box = obj.find('bndbox')

                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax])
            temp.append(np.array(label))
        return temp


