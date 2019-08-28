
import sys
import argparse
import subprocess as subp

def parse_args(arguments):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Debugging Transformer Compression on NGC")
    parser.add_argument('--command', help='sleep' )
    return parser.parse_args(arguments)
  
def call_ngc(name, command):
    ngc_command = [
        'ngc', 'batch', 'run', 
        '--name', name, 
        '--total-runtime', '168h',
        '--image', 'nvidian/pytorch:19.08-py3', 
        '--ace', 'nv-us-west-2',
        '--instance', 'dgx1v.16g.8.norm',
        '--commandline', 'sleep 168h', 
        '--result', '/results',
        '--datasetid', '13254:/imagenet',
        '--workspace', 'condensa-transformer:/condensa:RW',
        '--port', '8888',
    ]
    print(ngc_command)
    subp.check_call(ngc_command)

def main(arguments):
    args = parse_args(arguments)
    print(args)
    call_ngc('debug', args.command)

if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
  
