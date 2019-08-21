import condensa
from condensa  import schemes

import argparse
import sys

def parse_args(arguments):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Condensa Compression Script for Transformer on WikiText-2'
            )
    parser.add_argument('--arch', default='Transformer', help='Model Architecture')
    parser.add_argument('--dataset', default='wikitext2', type=str, help='Dataset')
    return parser.parse_args(arguments)

def main(arguments):
    args = parse_args(arguments)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
