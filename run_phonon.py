import argparse
import glob
import os
import subprocess
from shutil import copyfile, rmtree


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    args = parser.parse_args()

    if args.cell:
        src_dir = os.path.join(OPTIMISED_DIR, 'cell')
        dest_dir = os.path.join(TARGET_DIR, 'cell')
    else:
        src_dir = os.path.join(OPTIMISED_DIR, 'no_cell')
        dest_dir = os.path.join(TARGET_DIR, 'no_cell')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    data_files = sorted(glob.glob(os.path.join(src_dir, '*.vasp')))
    #print(data_files)
    
    for file in data_files:
        name = os.path.split(file)[-1].replace('.vasp', '')
        print(name)

        check_dir = os.path.join(src_dir, 'extra_data', name)
        if os.path.exists(os.path.join(check_dir, 'spacegroup_changed')):
            print(f'Skipping {name} because optimisation changed space group')
            continue
        elif not os.path.exists(os.path.join(check_dir, 'spacegroup_conserved')):
            if is_symmetry_broken(check_dir):
                print(f'Skipping {name} because optimisation changed space group')
                continue

        work_dir = os.path.join(dest_dir, name)
        
        if args.restart:
            rmtree(work_dir)
        else:
            if glob.glob(os.path.join(work_dir, '*force_constants*')):
                print('Skipping {name} because it is already complete')
                continue
            else:
                print(os.path.join(work_dir, '*force_constants'))
                rmtree(work_dir)

        os.makedirs(work_dir)
        os.chdir(work_dir)
        copyfile(file, os.path.join(work_dir, 'POSCAR'))

        base_args = ['janus', 'phonons', '--struct', './POSCAR', '--supercell', SUPERCELL,
            '--arch', 'mace_mp', '--model-path', 'large', '--calc-kwargs', '{"dispersion": True}',
            '--bands', '--dos', '--plot-to-file']
        
        try:
            result = subprocess.run(base_args + ['--device', 'cuda'], check=True)
        except subprocess.CalledProcessError:
            print('cuda run failed; retrying using CPU only')
            subprocess.run(base_args + ['--device', 'cpu'])

        os.chdir(HOME_DIR)
