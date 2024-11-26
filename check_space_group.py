import argparse
import glob
import os

from ase.io import read
from janus_core.calculations.single_point import SinglePoint
import numpy as np


TOP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.join(TOP_DIR, 'data', 'optimised')


def check_vesta(previous_file: str,
                current_name: str,
                arch: str,
                model_path: str,
                remove_dir: str,
                main_dir: str) -> None:
    previous_name = os.path.split(os.path.dirname(previous_file))[-1].replace('.vasp','')
    current_name = current_name.replace('.vasp', '')

    if not (previous_name in current_name or current_name in previous_name):
        return

    current_file = os.path.join(main_dir, current_name + '.vasp')
    previous_file = os.path.join(main_dir, previous_name + '.vasp')
    if previous_name[-6:] == '_vesta':
        vesta_file, vesta_name = previous_file, previous_name
        cif2cell_file, cif2cell_name = current_file, current_name
    elif current_name[-6:] == '_vesta':
        vesta_file, vesta_name = current_file, current_name
        cif2cell_file, cif2cell_name = previous_file, previous_name
    else:
        return

    energies, files, names = [], [cif2cell_file, vesta_file], [cif2cell_name, vesta_name]

    for file, name in zip(files, names):
        npy = os.path.join(os.path.dirname(file), 'extra_data', name, 'final.npy')
        if os.path.exists(npy):
            energies.append(np.load(npy)[0])
            continue

        atoms = read(file, format='vasp')
        sp = SinglePoint(struct=atoms,
                         arch=arch,
                         device='cuda',
                         model_path=model_path,
                         calc_kwargs={'dispersion': True},
                         properties='energy')

        result = sp.run()
        energies.append(result['energy'] / len(atoms))

    if energies[0] > energies[1]:
        print('VESTA file lower in energy')
        os.rename(cif2cell_file, os.path.join(remove_dir, cif2cell_name + '.vasp'))
    else:
        os.rename(vesta_file, os.path.join(remove_dir, vesta_name + '.vasp'))
        print('cif2cell file lower in energy')

    np.save(os.path.join(remove_dir, cif2cell_name + '.npy'), np.array(energies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-cc', '--check-changed', action='store_true',
                        help='Also check systems that have been put aside by optimise.py because it found that the '
                             'space group changed and therefore re-did the calculation using constraints.')
    args = parser.parse_args()

    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        target_dir = os.path.join(HOME_DIR, '_'.join([args.arch, p]))
    else:
        target_dir = os.path.join(HOME_DIR, '_'.join([args.arch, args.model_path]))

    if args.cell:
        target_dir = os.path.join(target_dir, 'cell')
    else:
        target_dir = os.path.join(target_dir, 'no_cell')

    duplicates_dir = os.path.join(target_dir, 'high_energy_structures')
    if not os.path.exists(duplicates_dir):
        os.makedirs(duplicates_dir)

    files = sorted(glob.glob(os.path.join(target_dir, 'extra_data', '*', '*-log.yml')))

    results = []
    previous_file = 'NONE'
    for file in files:
        if not args.check_changed and '_changed' in os.path.split(os.path.dirname(file))[-1]:
            continue

        before, after = None, None
        with open(file, 'r') as f:
            for line in f:
                if 'Before optimisation spacegroup:' in line:
                    before = line.split('Before optimisation spacegroup:')[-1].replace('"', '').strip()
                elif 'After optimization spacegroup:' in line:
                    after = line.split('After optimization spacegroup:')[-1].replace('"', '').strip()
                    break
            else: 
                raise Exception()

        ok = 'OK' if before == after else 'FAIL'
        name = os.path.split(os.path.dirname(file))[-1]
        print(f'{ok}    {name}   before: {before};  after: {after}')
        
        results.append(', '.join([name, str(before == after), before, after]))
        if before == after:
            title = 'spacegroup_conserved'
        else:
            title = 'spacegroup_changed'

        with open(os.path.join(os.path.dirname(file), title), 'w') as f:
            f.write(before + '   ' + after)

        check_vesta(previous_file, name, args.arch, args.model_path, duplicates_dir, target_dir)
        previous_file = file

    with open(os.path.join(target_dir, 'spacegroup_check.csv'), 'w') as f:
        f.writelines(results)
