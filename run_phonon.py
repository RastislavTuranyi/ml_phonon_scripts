"""
Script for running phonon calculations using janus and an MLIP.

Assumptions:

1.  The `optimise.py` script has been run with the same parameters and the results are available as
    expected.
2.  A 2x2x2 supercell is used.

This script computes phonons for all systems outputted by `optimise.py` where the symmetry remained
conserved from before optimisation. It includes the functionality for determining whether the
symmetry was conserved, but it can also work off of the `check_space_group.py` script, in which
case its outputs are used instead so as not to repeat I/O.

The janus phonon calculations are initiated using the janus CLI via `subprocess` and are run on a
GPU. However, should the calculations fail due to cuda/PyTorch running out of memory, the run is
retried with `PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'` environment variable in the
hopes that that might help, and should there still not be enough memory, the computation will
instead be run on the CPU, which will likely take significant resources.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from shutil import copyfile, rmtree

from ase.io import read, write
from ase.build.supercells import find_optimal_cell_shape
from ase.build import make_supercell
import numpy as np


HOME_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HOME_DIR, 'data')
OPTIMISED_DIR = os.path.join(DATA_DIR, 'optimised')
TARGET_DIR = os.path.join(HOME_DIR, 'results')

SUPERCELL = '2x2x2'
IDEAL_VOLUME = 16 ** 3
MINIMUM_CUTOFF = 7.5


class InvalidLogFile(Exception):
    pass


def is_symmetry_broken(path: str) -> bool:
    """
    Checks whether optimisation changed the symmetry of the space group by looking at yaml log
    file outputted by janus.

    :param path: Path to a directory containing the output of the `optimise.py` script for one system
    :return: Whether the symmetry changed during optimisation
    """
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
            raise InvalidLogFile('The janus log file is invalid: maybe the optimisation changed'
                                 ' or the spec changed in the latest janus version. Regardless,'
                                 ' the space group information could not be read.')
    return before != after


def has_symmetry_changed(src_dir: str, name: str) -> bool:
    """
    Checks whether optimisation changed the symmetry of a particular system. First, the flag files
    created by the `check_space_group.py` script are checked, and if they are not present, the
    janus log file is used to determine the symmetry change (see :py:func:`is_symmetry_broken`).

    :param src_dir: The path to the directory holding the results for all systems.
    :param name: The name of the system - this is the same name as the folder corresponding to the
                 system in `src_dir/extra_data`

    :return: Whether the symmetry changed
    """
    check_dir = os.path.join(src_dir, 'extra_data', name)
    if os.path.exists(os.path.join(check_dir, 'spacegroup_changed')):
        print(f'Skipping {name} because optimisation changed space group')
        return True
    elif os.path.exists(os.path.join(check_dir, 'spacegroup_conserved')):
        return False
    else:
        if is_symmetry_broken(check_dir):
            print(f'Skipping {name} because optimisation changed space group')
            return True

    return False


def is_calculation_complete(work_dir: str, name: str) -> bool:
    """
    Checks whether the calculation completed successfully (as indicated by the presence of a
    force constants file). If the calculation had been previously started and not finished, the
    entire directory is also deleted as a side effect.

    :param work_dir: The path to the directory containing the results for a given system
    :param name: The name of the system

    :return: Whether the calculation completed successfully
    """
    if glob.glob(os.path.join(work_dir, '*force_constants*')):
        print(f'Skipping {name} because it is already complete')
        return True

    try:
        rmtree(work_dir)
        print(f'Redoing {name} because previous calculation did not complete')
    except FileNotFoundError:
        pass

    return False


def get_supercell(path: str, work_dir: str, multiplier: float = 1.) -> str | None:
    """
    Constructs a supercell to use for the phonon calculations with janus.

    :param path: The path to the structure file to use.
    :return: The supercell as a string for input to janus phonons CLI.
    """
    atoms = read(path, format='vasp')
    geom_path = os.path.join(work_dir, 'geometry.in')
    write(geom_path, atoms)

    target_size = round(IDEAL_VOLUME / atoms.cell.volume)
    target_n = len(atoms.get_atomic_numbers()) * target_size * multiplier
    print(f'getting supercell: target={target_size} target n atom={target_n}')

    result = subprocess.run(['vibes', 'utils', 'make-supercell', 
                             geom_path, '-n', str(int(target_n))],
                            capture_output=True, encoding='utf8')

    for line in result.stdout.split('\n'):
        if 'cmdline:' in line:
            cell = line.replace('cmdline:', '').strip()
            break
    else:
        if 'LinAlgError' in result.stderr:
            return get_supercell(path, work_dir, multiplier + 0.5)
        print(result.returncode, result.stdout, result.stderr)
        raise Exception()

    for line in result.stdout.split('\n'):
        if 'Largest Cutoff' in line:
            cut_off = float(line.split()[-2])
            break
    else:
        raise Exception()

    for geometry in glob.glob(os.path.join(work_dir, '*.in')):
        os.remove(geometry)

    if cut_off < MINIMUM_CUTOFF:
        if target_n > 1000:
            print(f'FAILED!!!!!!!!!!!!!!!!   max_cutoff={cut_off}')
            return None

        cell = get_supercell(path, work_dir, multiplier + 0.5)

    return cell


def get_sc_supercell(cell: np.ndarray, target: int):
    metric = np.eye(3)
    norm = (target * abs(np.linalg.det(cell)) / np.linalg.det(metric)) ** (-1 / 3)
    norm_cell = norm * cell

    ideal = np.dot(metric, np.linalg.inv(norm_cell))
    return np.array(np.around(ideal, 0), dtype=int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-cs', '--check-supercells', action='store_true',
                        help='Disregards everything and only print out the supercell target sizes.')
    parser.add_argument('-rs', '--redo-supercells', action='store_true',
                        help='Redoes the supercells')
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

        if not args.check_supercells:
            if has_symmetry_changed(src_dir, name) or is_calculation_complete(work_dir, name):
               continue
        
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        try:
            copyfile(file, os.path.join(work_dir, 'POSCAR'))
        except FileExistsError:
            if not args.check_supercells:
                raise

        supercell_path = os.path.join(work_dir, 'supercell.npy')
        if os.path.exists(supercell_path) and not args.redo_supercells:
            supercell = np.load(supercell_path)
            print(f'supercell = {supercell}')
        else:
            supercell = get_supercell(file, work_dir)
            if supercell is None:
                continue
        
            print(f'supercell = {supercell}')
            np.save(supercell_path, np.array(supercell.split()).astype(int))

        if args.check_supercells:
            continue

        base_args = ['janus', 'phonons',
                     '--struct', './POSCAR',
                     '--supercell', supercell,
                     '--arch', args.arch,
                     '--model-path', args.model_path,
                     '--calc-kwargs', '{"dispersion": True}',
                     '--plot-to-file',
                     '--file-prefix', name,
                     '--no-tracker']
        # TODO: Compute bands but with no-write-mesh option

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
