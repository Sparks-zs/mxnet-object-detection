import os


class imdb(object):
    def __init__(self):
        self.num_images = 0
        self.image_set = None

    def image_path_from_index(self, index):
        raise NotImplementedError

    def label_from_index(self, index):
        raise NotImplementedError

    def image_shape_from_index(self, index):
        raise NotImplementedError

    def save_img_list(self, target_path, shuffle):
        """
        生成lst文件，保存指定路径中
        :param target_path: 目标路径
        """

        # 进度条
        def progress_bar(count, total, suffix=''):
            import sys
            bar_len = 24
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
            sys.stdout.flush()

        str_list = []
        for idx in range(self.num_images):
            # if idx == 100:
            #     break
            progress_bar(idx, self.num_images)
            label = self.label_from_index(idx)  # 图片类别和bbox标签
            img_shape = self.image_shape_from_index(idx)  # 图片宽高标签
            path = self.image_path_from_index(idx)  # 图片路径
            str_list.append('\t'.join([str(idx), str(4),
                                       str(label.shape[1]), str(img_shape[0]), str(img_shape[1])] +
                                      ['{0:.4f}'.format(x) for x in label.ravel()] + [path]) + '\n')

        if str_list:
            if shuffle:
                import random
                random.shuffle(str_list)

            fname = os.path.join(target_path, self.image_set + '.lst')
            with open(fname, 'w+') as f:  # 写入.lst文件中
                for line in str_list:
                    f.write(line)
        else:
            raise RuntimeError("No image in this file")