# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import math 
import os 
import torch 
import torch.nn as nn

import data
import model

import logging
import csv

import gzip
import pickle

import torch.nn.parallel
import torch.nn.utils
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import condensa
from condensa import schemes

import util
import model

def parse_args(arguments):

    valid_schemes = ['PRUNE', 'PQ', 'FILTER']
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='WikiTest-2 LC Compression Script')

    # positional arguments
    parser.add_argument('model_file', metavar='model', help='Pretrained model filename')
    parser.add_argument('scheme',
                        choices=valid_schemes,
                        help='Compression scheme')
    parser.add_argument('density',
                        type=float,
                        help='Density for pruning')
    parser.add_argument('steps', type=int, help='Number of LC iterations')

    # optional arguments
    parser.add_argument('--arch',
                        default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer' )
    parser.add_argument('--dataset', default='wikitext-2',
                        choices=('ptb', 'wikitext-2', 'wikitext-103'), type=str,
                        help='dataset name')

    parser.add_argument('--emsize',
                        type=int,
                        default=200,
                        help='size of word embeddings')

    parser.add_argument('--l_batch_size',
                        metavar='N',
                        type=int,
                        default=128,
                        help='Batch size for L step')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=100,
                        help='Validation batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.02,
                        help='Initial learning rate')
    parser.add_argument('--lr_end',
                        type=float,
                        default=None,
                        help='Ending learning rate')
    parser.add_argument('--lr_decay',
                        type=float,
                        default=1e-4,
                        help='Learning rate decay')
    parser.add_argument('--lr_schedule',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr_multiplier',
                        type=float,
                        default=None,
                        help='Learning rate multiplier')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.95,
                        help='SGD momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='SGD momentum')
    parser.add_argument('--mb_iterations_per_l',
                        type=int,
                        default=3000,
                        help='Minibatch iterations per L step')
    parser.add_argument('--mb_iterations_first_l',
                        type=int,
                        default=30000,
                        help='Minibatch iterations for first L step')
    parser.add_argument('--mu_init',
                        type=float,
                        default=0.001,
                        help='Initial value of mu')
    parser.add_argument('--mu_multiplier', type=float, default=1.1, help='mu multiplier')
    parser.add_argument('--mu_cap', type=float, default=10000, help='mu cap')
    parser.add_argument('--out',
                        default=None,
                        help='''
                            Compressed output model filename
                            (default=compressed/{model}_{scheme}_{density}.pth)
                            ''')
    parser.add_argument('--csv',
                        default=None,
                        help='''
                            compression statistics CSV file
                            (default=results/{model}_{scheme}_{density}.csv)
                            ''')
    parser.add_argument('-v',
                        '--verbose',
                        help='verbose logging output',
                        action='store_true')

    args = parser.parse_args(arguments)

    default_prefix = f'{args.model_file}_{args.scheme}_{args.density}'.replace('.', '_')
    if args.out is None:
        args.out = os.path.join('compressed', default_prefix + '.pth')
    if args.csv is None:
        args.csv = os.path.join('results', default_prefix + '.csv')

    if args.dataset == 'cifar10':
        args.dataset = datasets.CIFAR10
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.dataset = datasets.CIFAR100
        args.num_classes = 100
    else:
        parser.error('Invalid dataset: must be CIFAR-10 or CIFAR-100')

    args.arch_func = arch_map[args.arch]

    return args

def create_scheme(name):
    if args.scheme == 'PRUNE':
        scheme = schemes.Prune(args.density)
    elif args.scheme == 'PQ':
        scheme = schemes.Compose(
            [schemes.Prune(args.density),
             schemes.Quantize()])
    elif args.scheme == 'FILTER':
        scheme = schemes.FilterPrune(args.density)
    elif args.scheme == 'BLOCK':
        scheme = schemes.BlockPrune(args.density, (8,8))    
    else:
        # should never happen
        raise RuntimeError('Unknown scheme: {}'.format(args.scheme))
    return scheme

def create_compressor(args):
    trainloader, valloader = \
        util.cifar_train_val_loader(args.dataset,
                                    args.l_batch_size,
                                    args.val_batch_size)
    testloader = util.cifar_test_loader(args.dataset, args.val_batch_size)

    # Instantiate LC optimizer
    sgd_params = {'momentum': args.momentum, 'weight_decay': args.weight_decay}
    lc = condensa.opt.LC(steps=args.steps,
                         l_optimizer=condensa.opt.lc.SGD,
                         l_optimizer_params=sgd_params,
                         lr=args.lr,
                         lr_end=args.lr_end,
                         lr_decay=args.lr_decay,
                         lr_schedule=args.lr_schedule,
                         lr_multiplier=args.lr_multiplier,
                         mb_iterations_per_l=args.mb_iterations_per_l,
                         mb_iterations_first_l=args.mb_iterations_first_l,
                         mu_init=args.mu_init,
                         mu_multiplier=args.mu_multiplier,
                         mu_cap=args.mu_cap,
                         debugging_flags={'print_accuracies': True})

    criterion = nn.CrossEntropyLoss().cuda()
    # Compress model using Condensa
    compressor = condensa.Compressor(lc, scheme, model, trainloader,
                                     testloader, valloader, criterion)

    return compressor

def output_results(model_file, stats_file):
    torch.save(w.state_dict(), model_file)
    logging.info('[Condensa] Compressed model written to disk')

    logging.info('==== Profiling Results ====')
    for k, v in compressor.statistics.items():
        logging.info(f'{k}: {v}')

    with open(stats_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for k, v in compressor.statistics.items():
            row = [k]
            if isinstance(v, list): row += [str(x) for x in v]
            else: row.append(str(v))
            writer.writerow(row)
    logging.info('[Condensa] Compression stats written to disk')

def main(arguments):
    args = parse_args(arguments)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s')

    model = args.arch_func(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_file))

    scheme = create_scheme(args.scheme)
    logging.info(f'SCHEME: {scheme}')

    compressor = create_compressor(args)
    w = compressor.run()

    output_results(args.out, args.csv)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
