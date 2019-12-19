

import os,sys
import argparse


def get_args():
    """
    Parses arguments passed in command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default = './models/'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default=''
    )
    parser.add_argument(
        '--load_best',
        dest = 'load_best',
        action= 'store_true'
    )
    parser.add_argument(
        '--train',
        dest = 'load_best',
        action= 'store_false'
    )
    parser.set_defaults(load_best=True)
    args, _ = parser.parse_known_args()
    return args

