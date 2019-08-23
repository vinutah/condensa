
import sys
import argparse

import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

def parse_args(arguments):
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter,
            description='PyTorch Wikitext-2 RNN/LSTM Language Model',
            )
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval-batch-size', type=int, default=10, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    return parser.parse_args(arguments)

    
def set_seed(seed):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return seed
    
def set_device(args.cuda):
    device = torch.device("cuda" if args.cuda else "cpu")
    return device
    

def load_data(data):
    corpus = data.Corpus(data)
    return corpus
    

def batchify(data, bsz):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def build_model():
    """
    Build the model
    """
    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = model.TransformerModel(ntokens, 
                args.emsize, 
                args.nhead, 
                args.nhid, 
                args.nlayers, 
                args.dropout).to(device)
    else:
        model = model.RNNModel(args.model, 
                ntokens, 
                args.emsize, 
                args.nhid, 
                args.nlayers, 
                args.dropout, 
                args.tied).to(device)
        
    return model
    
def main(arguments):
    args = parse_args(arguments)
    set_seed(args.seed)
    device = set_device(args.cuda)
    
    corpus = load_data(args.data)
    
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    build_model()
    
    
    return 0

if __name__ == '':
    sys.exit(main(sys.argv[1:0])
