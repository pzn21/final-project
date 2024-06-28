import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data', help='path of dataset')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--lr-gen', type=float, default=2e-4, help='learning rate of the generator')
parser.add_argument('--lr-dis', type=float, default=1e-4, help='learning rate of the discriminator')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--device', type=str, default='cuda', help='running device')
args = parser.parse_args()