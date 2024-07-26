import pyarrow.parquet as pq
import re
import sys
import os
import argparse
from tqdm import tqdm
from dlomix.constants import ALPHABET_NAIVE_MODS, ALPHABET_UNMOD, PTMS_ALPHABET


def create_parser():
    parser = argparse.ArgumentParser(
        prog='GetAlphabet',
        description='Give a parquet file, and optionally an alphabet you want to update and recieve the updated alphabet'
    )

    parser.add_argument(
        '-p', 
        '--parquet',
        help='The dataset with .parquet ending you want to parse'
        )
    
    parser.add_argument(
        '-a',
        '--alphabet',
        default='unmod',
        required=False,
        help='Can specify an alphabet in "unmod", "naive" or "ptm" and recieve the updated version (Defaults to umod)'
    )

    return parser.parse_args()


def get_modification(seq):
    mod_pattern = r"[A-Za-z]?(?:\[UNIMOD:\d+\])*|[^\[\]]"
    splitted = seq.split('-')

    match len(splitted):
        case 1:
            n_term, seq, c_term = '[]-', splitted[0], '-[]'
        case 2:
            if splitted[0].startswith('[UNIMOD:'):
                n_term, seq, c_term = splitted[0] + '-', splitted[1], '-[]'
            else:
                n_term, seq, c_term = '[]-', splitted[0], '-' + splitted[1]
        case 3:
            n_term, seq, c_term = splitted[0] + '-', splitted[1], '-' + splitted[2]
    
    seq = re.findall(mod_pattern, seq)
    seq.extend([n_term, c_term])
    return seq



def main():
    args = create_parser()
    if args.alphabet not in ['unmod', 'naive', 'ptm', 'new']:
        sys.exit('Specify a valid alphabet you want to update!\nChoose between, unmod, naive, ptm or new.')
    else:
        match args.alphabet:
            case 'unmod':
                alphabet = ALPHABET_UNMOD
            case 'naive':
                alphabet = ALPHABET_NAIVE_MODS
            case 'ptm':
                alphabet = PTMS_ALPHABET
            case _:
                alphabet = dir()

    if not args.parquet.endswith('.parquet'):
        sys.exit('File specified is not a .parquet file! Please specify a file with the .parquet ending.')

    modifications = set()

    file = pq.ParquetFile(args.parquet)
    print(f'Start processing dataset: {os.path.basename(args.parquet)}\n')
    # iterate over batches of the file
    total_seqs = 0
    for batch in tqdm(file.iter_batches()):
        total_seqs += len(batch)
        for cur_seq in batch['modified_sequence']:
            cur_mods = get_modification(str(cur_seq))
            modifications |= set(cur_mods)
    
    # get modifications not present in the old alphabet
    new_mods = modifications - set(alphabet.keys()) 
    # remove '' if in new mods
    try:
        new_mods.remove('')
    except KeyError as e:
        pass
    print(f'Total sequences in this dataset {total_seqs = }\n')
    
    print(f'These modifications did not occur in the specified alphabet: ')
    print(', '.join(list(new_mods)), '\n')

    # update alphabet with new values
    new_alphabet = {mod: v for v, mod in enumerate(new_mods, start=len(alphabet) + 1)}

    alphabet.update(new_alphabet)
    print('The updated alphabet is: ')
    print(alphabet)


if __name__ == '__main__':
    main()
