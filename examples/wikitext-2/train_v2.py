
import sys
import argparse

import time
import math
import os

import torch
import torch.nn as nn
import torch.onnx

import data
import model as model_module

def parse_args(arguments):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='PyTorch Wikitext-2 RNN/LSTM Language Model',
            )
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        choices = ('RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU',
                                   'Transformer'),
                        help='type of recurrent net')
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
                        help='''
                            the number of heads in the encoder/decoder of the
                            transformer model.
                            ''')
    args = parser.parse_args(arguments)
    print(args)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run'
                  'with --cuda')
    return args

def batchify(data, batch_size, device):
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
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def load_model(filename, model_type):
    # Load the best saved model.
    with open(filename, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()
    return model

def build_model(args, ntokens, device):
    """
    Build the model
    """
    if args.model == 'Transformer':
        model = model_module.TransformerModel(
            ntokens,
            args.emsize,
            args.nhead,
            args.nhid,
            args.nlayers,
            args.dropout,
        ).to(device)
    else:
        model = model_module.RNNModel(
            args.model,
            ntokens,
            args.emsize,
            args.nhid,
            args.nlayers,
            args.dropout,
            args.tied,
        ).to(device)

    return model


def train(args, model, ntokens, train_loader, criterion, lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, data_and_targets in enumerate(train_loader):
        data, targets = data_and_targets
        # Starting each batch, we detach the hidden state from how it was
        # previously produced.  If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                batch, (len(train_loader) + 1) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(args, model, data_loader, ntokens, criterion):
    model.eval() # Turn on evaluation mode which disables dropout.
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for data, targets in data_loader:
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / len(data_loader)

class BatchLoader:
    def __init__(self, data, bptt):
        self.data = data
        self.bptt = bptt
    def __len__(self):
        return len(self.data) - 1
    def __iter__(self):
        '''
        get_batch subdivides the source data into chunks of length
        args.bptt.  If source is equal to the example output of the
        batchify function, with a bptt-limit of 2, we'd get the
        following two Variables for i = 0:

        ┌ a g m s ┐ ┌ b h n t ┐
        └ b h n t ┘ └ c i o u ┘

        Note that despite the name of the function, the subdivison of
        data is not done along the batch dimension (i.e. dimension 1),
        since that was handled by the batchify function. The chunks
        are along dimension 0, corresponding to the seq_len dimension
        in the LSTM.
        '''
        for i in range(0, len(self), self.bptt):
            seq_len = min(self.bptt, len(self) - i)
            data = self.data[i:i+seq_len]
            target = self.data[i+1:i+1+seq_len].view(-1)
            yield data, target

def make_data_loaders(corpus, device, train_batch_size, eval_batch_size, bptt):
    'Returns three loaders: (train_loder, val_loader, test_loader)'
    train_data = batchify(corpus.train, train_batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    train_loader = BatchLoader(train_data, bptt)
    val_loader = BatchLoader(val_data, bptt)
    test_loader = BatchLoader(test_data, bptt)

    return train_loader, val_loader, test_loader

def run_training(trainer, model, evaluator, train_loader, val_loader,
                 lr, epochs, filename):
    best_val_loss = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        trainer(train_loader, lr)
        val_loss = evaluator(model, val_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so
        # far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(filename, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in
            # the validation dataset.
            lr /= 4.0

def main(arguments):
    args = parse_args(arguments)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    train_loader, val_loader, test_loader = make_data_loaders(
        corpus, device, args.batch_size, args.eval_batch_size, args.bptt)

    model = build_model(args, ntokens, device)
    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.

    trainer = lambda train_loader, learning_rate: \
        train(args, model, ntokens, train_loader, criterion,
              learning_rate)
    evaluator = lambda model, data_loader: \
        evaluate(args, model, data_loader, ntokens, criterion)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        run_training(trainer, model, evaluator, train_loader, val_loader,
                     args.lr, args.epochs, args.save)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model = load_model(args.save, args.model)
    test_loss = evaluator(model, test_loader)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
    print('=' * 89)

    #if len(args.onnx_export) > 0:
    #    # Export the model in ONNX format.
    #    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
