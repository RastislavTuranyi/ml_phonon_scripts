import argparse
import glob
import os

from ase.io import read
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OPTIMISED_DIR = os.path.join(DATA_DIR, 'optimised')
START_DIR = os.path.join(DATA_DIR, 'primitive')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    args = parser.parse_args()

    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        result_dir = os.path.join(OPTIMISED_DIR, '_'.join([args.arch, p]))
    else:
        result_dir = os.path.join(OPTIMISED_DIR, '_'.join([args.arch, args.model_path]))

    result_dir = os.path.join(result_dir, 'cell') if args.cell else os.path.join(result_dir, 'no_cell')
    extra_dir = os.path.join(result_dir, 'extra_data')

    files = sorted(glob.glob(os.path.join(OPTIMISED_DIR, '*.vasp')))
    for file in files:
        print('\n')
        name = os.path.split(file)[-1]
        print(name)

        optimised = read(file, format='vasp').cell.cellpar()
        primitive = read(os.path.join(START_DIR, name), format='vasp').cell.cellpar()

        if np.allclose(primitive, optimised):
            print('OK')
        else:
            print(f'FAIL; primitive={primitive},   optimised={optimised}')
