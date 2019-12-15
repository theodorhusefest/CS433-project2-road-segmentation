
""" Contains all basic helperfunctions, e.g load_image()"""

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
    args, _ = parser.parse_known_args()
    return args

