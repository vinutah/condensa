
import data
import model

import condensa
from condensa  import schemes

import sys
import argparse
import logging

import torch
import torch.nn as nn

import csv



def parse_args(arguments):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Condensa Compression Script for Transformer on WikiText-2'
            )
    # Model related
    parser.add_argument('--arch', default='Transformer', help='Model Architecture')
    parser.add_argument('--model', default='./trained/transformer.pth', help='Pretrained model filename')

    # Transformer Model related
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=20, metavar='N', help='eval batch size')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    #parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    #parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    #parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the encoder/decoder of the transformer model')
    
    # Dataset related
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset')

    # Condensa related
    parser.add_argument('--steps', type=int, help='Number of LC iterations')
    parser.add_argument('--lr', type=float, default=0.5, help='Initial learning rate')
    parser.add_argument('--lr_end', type=float, default=None, help='Ending learning rate')
    parser.add_argument('--lr_decay', type=float, default=None, help='Learning rate decay')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=None, help='Decrease learning rates at these epochs')
    parser.add_argument('--lr_multiplier', type=float, default=None, help='Learning rate multiplier')
   
    # Condensa Compression Program related
    valid_schemes = ['PRUNE', 'PQ', 'FILTER']
    parser.add_argument('--scheme', choices=valid_schemes, required=True, help='Compression scheme')
    parser.add_argument('--density', required=True, type=float, help='Density for pruning')

    # SGD reated TODO might not be required ?
    parser.add_argument('--weight_decay', type=float, default=0, help='sgd momentum')
    parser.add_argument('--momentum', type=float, default=0.95, help='sgd momentum')

    # Condensa L Step Related
    parser.add_argument('--l_batch_size', type=int, default=128, help='Batch size for L step')
    parser.add_argument('--val_batch_size', type=int, default=100, help='Validation batch size')

    # Condensa LC optimizer related
    parser.add_argument('--mb_iterations_per_l', type=int, default=2000, help='Minibatch iterations per L step')
    parser.add_argument('--mb_iterations_first_l', type=int, default=10000, help='Minibatch iterations for first L step')
    parser.add_argument('--mu_init', type=float, default=0.001, help='Initial value of mu')
    parser.add_argument('--mu_multiplier', type=float, help='mu multiplier')
    parser.add_argument('--mu_cap', type=float, default=10000, help='mu cap')
    
    # Compression Outputs
    parser.add_argument('--out', default='compressed_model.pth', help='Compressed output model filename')
    parser.add_argument('--csv', default=None, help='compression statistics CSV file')
    parser.add_argument('-v', '--verbose', help='verbose logging output', action='store_true')


    return parser.parse_args(arguments)

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
    >>> train_data = batchify(corpus.train, train_batch_size, device)
    >>> val_data = batchify(corpus.valid, eval_batch_size, device)
    >>> test_data = batchify(corpus.test, eval_batch_size, device)
    """
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

class BatchLoader:
    """
    subdivides the source data into chunks of length
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
    >>> train_loader = BatchLoader(train_data, bptt)
    >>> val_loader   = BatchLoader(val_data, bptt)
    >>> test_loader  = BatchLoader(test_data, bptt)
    """
    def __init__(self, data, bptt):
        self.data = data
        self.bptt = bptt
    def __len__(self):
        return len(self.data) - 1
    def __iter__(self):
        for i in range(0, len(self), self.bptt):
            seq_len = min(self.bptt, len(self) - i)
            data = self.data[i:i+seq_len]
            target = self.data[i+1:i+1+seq_len].view(-1)
            yield data, target

def make_data_loaders(corpus, device, train_batch_size, eval_batch_size, bptt):
        """
        Returns three loaders: (train_loder, val_loader, test_loader)
        >>> train_loader, val_loader, test_loader = make_data_loaders( corpus, device,
                                                                       args.batch_size, 
                                                                       args.eval_batch_size, 
                                                                       args.bptt)
        """
        train_data = batchify(corpus.train, train_batch_size, device)
        val_data = batchify(corpus.valid, eval_batch_size, device)
        test_data = batchify(corpus.test, eval_batch_size, device)

        train_loader = BatchLoader(train_data, bptt)
        val_loader = BatchLoader(val_data, bptt)
        test_loader = BatchLoader(test_data, bptt)
        
        return train_loader, val_loader, test_loader 

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

def create_compressor(args,model,corpus,device, ntokens):
    eval_batch_size = 10

    # lc: Instantiate LC optimizer
    sgd_params = {'momentum': args.momentum, 'weight_decay': args.weight_decay, 'grad_clip': args.clip}
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

    # scheme: Setup Compression Scheme
    scheme = create_scheme(args.scheme)
    logging.info(f'SCHEME: {scheme}')
   
    # 3 dataloaders
    train_loader, val_loader, test_loader = make_data_loaders(corpus, device, args.batch_size, args.eval_batch_size, args.bptt)
    
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Compress model using Condensa
    compressor = condensa.Compressor(lc, scheme, model, train_loader,
                                     test_loader, val_loader, criterion, ntokens)

    return compressor

def output_results(w, compressor, model_file, stats_file):
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
    
    device = torch.device("cuda" if args.cuda else "cpu")
    corpus = data.Corpus(args.dataset)
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s')
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Load dataset
    corpus = data.Corpus(args.dataset)
    
    # Load model architecture
    ntokens = len(corpus.dictionary)
    print(args.arch)
    if args.arch == 'Transformer':
        import model
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
        print(model)

    # Load pre-trained model
    with open(args.model, 'rb') as f:
        model = torch.load(f).to(device)
    
    # Condensa: Run Compression
    compressor = create_compressor(args,model,corpus,device, ntokens)
    w = compressor.run()

    # Condensa: Output Results
    output_results(w, compressor, args.out, args.csv)
    return 0

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    sys.exit(main(sys.argv[1:]))
