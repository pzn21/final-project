import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data', help='path of dataset')
args = parser.parse_args()