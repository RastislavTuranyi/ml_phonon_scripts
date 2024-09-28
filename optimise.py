import argparse
import glob
import os
from shutil import copyfile, rmtree
import subprocess


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SOURCE_DIR = os.path.join(DATA_DIR, 'primitive')
TARGET_DIR = os.path.join(DATA_DIR, 'optimised')

FMAX = 1e-6


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
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, p]))
    else:
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, args.model_path]))

    if args.cell:
        cell = '--opt-cell-lengths'
        target_dir = os.path.join(target_dir, 'cell')
    else:
        cell = '--no-opt-cell-lengths'
        target_dir = os.path.join(target_dir, 'no_cell')
    
    extra_dir = os.path.join(target_dir, 'extra_data')
    if not os.path.exists(extra_dir):
        os.makedirs(extra_dir)

    files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.vasp')))

    for file in files:
        name = os.path.split(file)[-1]

        out_path = os.path.join(target_dir, name)
        out_dir = os.path.join(extra_dir, name.replace('.vasp', ''))

        if os.path.exists(out_path):
            if args.restart:
                print(f'Redoing {name} from scratch')
                rmtree(out_path)
                rmtree(out_dir)
            else:
                print(f'Skipping {name} because it is already complete')
                continue

        os.makedirs(out_dir)
        copyfile(file, os.path.join(out_dir, 'POSCAR'))
        os.chdir(out_dir)

        subprocess.run(['janus', 'geomopt',
                        '--struct', './POSCAR',
                        '--arch', args.arch,
                        '--device', 'cuda',
                        '--out', out_path,
                        '--calc-kwargs', '{"dispersion": True}',
                        '--model-path', args.model_path,
                        cell,
                        '--fmax', str(FMAX)])

        os.chdir(DATA_DIR)
        print('FINISHED')
