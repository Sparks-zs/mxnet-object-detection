from tools.imdb import imdb
import random

class ConcatDB(imdb):
    def __init__(self, imdbs, image_set, shuffle):
        """
        连接多个imdb
        :param imdbs:   imdb列表
        :param image_set: 数据集属性 train,val,trianval,test
        :param shuffle: 是否打乱顺序
        """
        super(ConcatDB, self).__init__()
        self.imdbs = imdbs
        self.image_set = image_set
        self.image_index = self._load_image_index(shuffle)      # 索引

    def _load_image_index(self, shuffle):
        """

        :param shuffle: 是否打乱顺序
        :return: 返回索引列表
        """
        self.num_images = 0
        for db in self.imdbs:
            self.num_images += db.num_images
        indices = list(range(self.num_images))
        if shuffle:
            random.shuffle(indices)
        return indices

    def _locate_index(self, index):
        """

        :param index: 图片索引
        :return: 返回imdb对象，图片真实索引
        """
        assert index >= 0 and index < self.num_images, "index out of range"
        pos = self.image_index[index]
        for k, v in enumerate(self.imdbs):
            if pos >= v.num_images:
                pos -= v.num_images
            else:
                return (k, pos)

    def image_path_from_index(self, index):
        """
        当前图片路径
        :param index: 图片索引
        :return: 当前图片路径
        """
        assert self.image_index is not None, 'Dataset not initialized'
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].image_path_from_index(n_index)

    def label_from_index(self, index):
        """
        当前图片label
        :param index:   图片索引
        :return:    图片label
        """
        assert self.image_index is not None, 'Dataset not initialized'
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].label_from_index(n_index)

    def image_shape_from_index(self, index):
        """
        当前图片宽高
        :param index:   图片索引
        :return:    图片宽高
        """
        assert self.image_index is not None, 'Dataset not initialized'
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].image_shape_from_index(n_index)
