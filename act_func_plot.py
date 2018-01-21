#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'Chainernの活性化関数を可視化する'
#

import argparse
import numpy as np
import matplotlib.pyplot as plt

import chainer.functions as F

from func import argsPrint, getFilePath


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='活性化関数の図を保存するフォルダ (default: ./result/)')

    return parser.parse_args()


def main(args):
    actfun = [
        F.relu, F.elu, F.clipped_relu, F.leaky_relu, F.selu,
        F.sigmoid, F.hard_sigmoid, F.tanh, F.softplus,
    ]
    x = np.arange(-10, 10, 0.2)

    for af in actfun:
        file_name = getFilePath(args.out_path, af.__name__, '.png')
        print(file_name)
        y = [i.data for i in af(x)]
        f = plt.figure()
        a = f.add_subplot(111)
        a.plot(x, np.array(y))
        plt.savefig(file_name, dpi=200)


if __name__ == '__main__':
    args = command()
    argsPrint(args)
    main(args)
