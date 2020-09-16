import os, sys
import mxnet
import argparse
import subprocess
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from tools.pascal_voc import PascalVoc
from tools.concat_db import ConcatDB


def load_pascal(devkit_path, target_path, image_set, year, class_names, shuffle):
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    for s in image_set:
        imdbs = []
        for y in year:
            imdbs.append(PascalVoc(devkit_path, s, y, class_names, shuffle))
        if len(imdbs) > 1:
            ConcatDB(imdbs, s, shuffle).save_img_list(target_path, shuffle)
        else:
            imdbs[0].save_img_list(target_path, shuffle)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lst and rec for dataset')
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=None, type=str)
    parser.add_argument('--target', dest='target_path', help='output list path',
                        default=None, type=str)
    parser.add_argument('--set', dest='set', help='trainval, train, trainval, val or test',
                        default='train,val', type=str)
    parser.add_argument('--year', dest='year', help='voc year',
                        default='2007,2012', type=str)
    parser.add_argument('--class-names', dest='class_names', help='choice class to use',
                        default='dog', type=str)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list',
                        default=True, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("saving list to disk...")
    load_pascal(args.root_path, args.target_path, args.set, args.year, args.class_names, args.shuffle)
    print('{} list file {} is generated ...'.format(args.set, args.target_path))

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    subprocess.check_call(['python', im2rec_path,
                           os.path.abspath(args.target_path),
                           os.path.abspath(args.root_path),
                           '--pack-label'])
    print('Record file  is generated ...')

    # example:
    # python tools/prepare_dataset.py --root data/VOCdevkit --target data/miniVOCdevkit/dog
    # load_pascal('D:/mxnet_projects/mxnet_ssd/data/VOCdevkit','D:/mxnet_projects/mxnet_ssd/data/miniVOCdevkit',
    #                  'train, val','2007,2012','dog', True)
