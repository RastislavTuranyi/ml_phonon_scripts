import argparse
import glob
import os
import subprocess
from shutil import copyfile, rmtree

from ase.io import read
import numpy as np


HOME_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HOME_DIR, 'data')
OPTIMISED_DIR = os.path.join(DATA_DIR, 'optimised')
TARGET_DIR = os.path.join(HOME_DIR, 'results')

SUPERCELL = '2x2x2'


def is_symmetry_broken(path: str) -> bool:
    file = glob.glob(os.path.join(path, '*-log.yml'))[0]
    
    before, after = None, None
    with open(file, 'r') as f:
        for line in f:
            #print(line)
            if 'Before optimisation spacegroup:' in line:
                before = line.split('Before optimisation spacegroup:')[-1].replace('"', '').strip()
            elif 'After optimization spacegroup:' in line:
                after = line.split('After optimization spacegroup:')[-1].replace('"', '').strip()
                break
            else: 
                raise Exception()
    return before == after


def has_symmetry_changed(src_dir: str, name: str) -> bool:
    check_dir = os.path.join(src_dir, 'extra_data', name)
    if os.path.exists(os.path.join(check_dir, 'spacegroup_changed')):
        print(f'Skipping {name} because optimisation changed space group')
        return True
    elif not os.path.exists(os.path.join(check_dir, 'spacegroup_conserved')):
        if is_symmetry_broken(check_dir):
            print(f'Skipping {name} because optimisation changed space group')
            return True

    return False


def is_calculation_complete(work_dir: str, name: str) -> bool:
    if glob.glob(os.path.join(work_dir, '*force_constants*')):
        print(f'Skipping {name} because it is already complete')
        return True

    try:
        rmtree(work_dir)
        print(f'Redoing {name} because previous calculation did not complete')
    except FileNotFoundError:
        pass

    return False


def get_supercell(path: str) -> str:
    atoms = read(path, format='vasp')

    cell_lengths = atoms.cell.cellpar()[:3]
    return 'x'.join(['1' if length > 20 else '2' for length in cell_lengths])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    args = parser.parse_args()

    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        src_dir = os.path.join(OPTIMISED_DIR, '_'.join([args.arch, p]))
        dest_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, p]))
    else:
        src_dir = os.path.join(OPTIMISED_DIR, '_'.join([args.arch, args.model_path]))
        dest_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, args.model_path]))

    if args.cell:
        src_dir = os.path.join(src_dir, 'cell')
        dest_dir = os.path.join(dest_dir, 'cell')
    else:
        src_dir = os.path.join(src_dir, 'no_cell')
        dest_dir = os.path.join(dest_dir, 'no_cell')

    if args.restart:
        rmtree(dest_dir)
        os.makedirs(dest_dir)
    else:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    data_files = sorted(glob.glob(os.path.join(src_dir, '*.vasp')))
    #print(data_files)
    
    for file in data_files:
        name = os.path.split(file)[-1].replace('.vasp', '')
        work_dir = os.path.join(dest_dir, name)
        print(name)

        if has_symmetry_changed(src_dir, name) or is_calculation_complete(work_dir, name):
            continue

        os.makedirs(work_dir)
        os.chdir(work_dir)
        copyfile(file, os.path.join(work_dir, 'POSCAR'))

        base_args = ['janus', 'phonons',
                     '--struct', './POSCAR',
                     '--supercell', get_supercell(file),
                     '--arch', args.arch,
                     '--model-path', args.model_path,
                     '--calc-kwargs', '{"dispersion": True}',
                     '--bands', '--plot-to-file',
                     '--file-prefix', name]
        
        try:
            result = subprocess.run(base_args + ['--device', 'cuda'], check=True)
        except subprocess.CalledProcessError:
            print('cuda run failed; retrying using PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
            try:
                env = os.environ.copy()
                env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                result = subprocess.run(base_args + ['--device', 'cuda'], check=True, env=env)
            except subprocess.CalledProcessError:
                print('cuda run failed again; retrying using CPU only')
                subprocess.run(base_args + ['--device', 'cpu'])

        os.chdir(HOME_DIR)

    print('FINISHED')
