import argparse
import glob
import os
from shutil import copyfile, rmtree
import subprocess

import ase
from ase.constraints import FixSymmetry
from ase.io import read, write

from janus_core.calculations.geom_opt import GeomOpt
from janus_core.calculations.single_point import SinglePoint

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SOURCE_DIR = os.path.join(DATA_DIR, 'primitive')
TARGET_DIR = os.path.join(DATA_DIR, 'optimised')

FMAX = 1e-6


def recompute_changed(original_file: str,
                      original_dir: str,
                      original_name: str,
                      source_file: str,
                      cell: bool,
                      arch: str,
                      model_path: str,
                      fkwargs: dict):
    # Move output file to extra data dir and rename the dir
    os.rename(original_file, os.path.join(original_dir, original_name))
    os.rename(original_dir, original_dir + '_changed')

    os.makedirs(original_dir)
    os.chdir(original_dir)

    atoms = read(source_file, format='vasp')
    atoms.set_constraint(FixSymmetry(atoms=atoms, adjust_positions=True, adjust_cell=cell))
    optimiser = GeomOpt(struct=atoms,
                        arch=arch,
                        device='cuda',
                        model_path=model_path,
                        calc_kwargs={'dispersion': True},
                        attach_logger=True,
                        fmax=FMAX,
                        write_results=True,
                        filter_kwargs=fkwargs)
    optimiser.run()
    final_force = np.linalg.norm(optimiser.struct.get_forces(), axis=1).max()

    if optimiser.struct.info['initial_spacegroup'] == optimiser.struct.info['final_spacegroup']:
        title = 'spacegroup_conserved'
    else:
        print('Space group changed despite ASE constraint')
        title = 'spacegroup_changed'

    write(original_file, optimiser.struct, format='vasp')

    with open(os.path.join(original_dir, title), 'w') as f:
        f.write(optimiser.struct.info['initial_spacegroup'] + '   ' + optimiser.struct.info['final_spacegroup'])

    np.save(os.path.join(original_dir, 'final.npy'),
            np.array([optimiser.struct.get_potential_energy(), final_force]))

    return final_force


def check_cif2cell_vesta(save_dir: str, top_dir: str, arch: str, model_path: str):
    print('Comparing cif2cell and vesta files')
    vesta_files = glob.glob(os.path.join(save_dir, '*_vesta.vasp'))

    duplicates_dir = os.path.join(top_dir, 'high_energy_structures')
    if not os.path.exists(duplicates_dir):
        os.makedirs(duplicates_dir)

    for vesta_file in vesta_files:
        cif2cell_file = vesta_file.replace('_vesta.vasp', '.vasp')
        cif2cell_name = os.path.split(cif2cell_file)[-1].replace('.vasp', '')

        print(cif2cell_name)
        if not os.path.exists(cif2cell_file):
            print('Skipping because equivalent cif2cell file does not exist (might have previously been moved)')
            continue

        vesta_name = os.path.split(vesta_file)[-1].replace('.vasp', '')
        vesta_dir = os.path.join(vesta_file.replace('.vasp', ''), 'extra_data', vesta_name)

        vesta_energy = os.path.join(vesta_dir, 'final.npy')
        if os.path.exists(vesta_energy):
            vesta_energy = np.load(vesta_energy)[0]
        else:
            vesta_energy = compute_one_energy(vesta_file, arch, model_path)

        cif2cell_dir = os.path.join(cif2cell_file.replace('.vasp', ''), 'extra_data', cif2cell_name)
        cif2cell_energy = os.path.join(cif2cell_dir, 'final.npy')
        if os.path.exists(cif2cell_energy):
            cif2cell_energy = np.load(cif2cell_energy)[0]
        else:
            cif2cell_energy = compute_one_energy(cif2cell_file, arch, model_path)

        if cif2cell_energy > vesta_energy:
            print('VESTA file lower in energy')
            os.rename(cif2cell_file, os.path.join(duplicates_dir, cif2cell_name + '.vasp'))
        else:
            os.rename(vesta_file, os.path.join(duplicates_dir, vesta_name + '.vasp'))
            print('cif2cell file lower in energy')


def compute_one_energy(file_path: str, arch: str, model_path: str) -> float:
    atoms = read(file_path, format='vasp')
    sp = SinglePoint(struct=atoms,
                     arch=arch,
                     device='cuda',
                     model_path=model_path,
                     calc_kwargs={'dispersion': True},
                     properties='energy')

    result = sp.run()
    return result['energy'] / len(atoms)


def run_geometry_optimisation(atoms: ase.Atoms, arch: str, model_path: str, filter_kwargs: dict):
    optimiser = GeomOpt(struct=atoms,
                        arch=arch,
                        device='cuda',
                        model_path=model_path,
                        calc_kwargs={'dispersion': True},
                        attach_logger=True,
                        fmax=FMAX,
                        write_results=True,
                        filter_kwargs=filter_kwargs)
    optimiser.run()
    return optimiser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    args = parser.parse_args()

    filter_kwargs = {'hydrostatic_strain': args.cell, 'scalar_pressure': 0.}

    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, p]))
    else:
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, args.model_path]))

    target_dir = os.path.join(target_dir, 'cell') if args.cell else os.path.join(target_dir, 'no_cell')

    if args.restart:
        rmtree(target_dir)
        os.makedirs(target_dir)
    
    extra_dir = os.path.join(target_dir, 'extra_data')
    if not os.path.exists(extra_dir):
        os.makedirs(extra_dir)

    files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.vasp')))

    not_converged, changed_despite_constraint = [], []
    for file in files:
        name = os.path.split(file)[-1]
        print(name)

        out_path = os.path.join(target_dir, name)
        out_dir = os.path.join(extra_dir, name.replace('.vasp', ''))

        if os.path.exists(out_path):
            print(f'Skipping {name} because it is already complete')

            if os.path.exists(os.path.join(out_dir, 'spacegroup_changed')):
                print('Recomputing data because space group in the original was changed')
                final_force = recompute_changed(out_path, out_dir, name, file, args.cell, args.arch, args.model_path,
                                                filter_kwargs)
                if final_force > FMAX:
                    print('WARNING: Constrained optimisation did not converge')
                    not_converged.append(name)
            continue
        elif os.path.exists(out_dir):
            if os.path.exists(os.path.join(target_dir, 'high_energy_structures', name)):
                print('Skipping because the structure is already complete and has been placed to high_energy_structures')
                continue

            print('Previously, optimisiation was started but not finished; starting over')
            rmtree(out_dir)

        os.makedirs(out_dir)
        copyfile(file, os.path.join(out_dir, name))
        os.chdir(out_dir)

        atoms = read(file, format='vasp')
        optimiser = run_geometry_optimisation(atoms, args.arch, args.model_path, filter_kwargs)
        energy = optimiser.struct.get_potential_energy()

        if optimiser.struct.info['initial_spacegroup'] != optimiser.struct.info['final_spacegroup']:
            print('Space group changed during optimisation -> retrying with fixed symmetry')

            atoms = read(file, format='vasp')
            atoms.set_constraint(FixSymmetry(atoms=atoms, adjust_positions=True, adjust_cell=args.cell))
            optimiser = run_geometry_optimisation(atoms, args.arch, args.model_path, filter_kwargs)

            energy2 = optimiser.struct.get_potential_energy()
            print(f'Original energy: {energy}; new energy: {energy2}')
            energy = energy2

            if optimiser.struct.info['initial_spacegroup'] == optimiser.struct.info['final_spacegroup']:
                title = 'spacegroup_conserved'
            else:
                print('Space group changed despite ASE constraint')
                title = 'spacegroup_changed'
                changed_despite_constraint.append(name)
        else:
            title = 'spacegroup_conserved'

        write(out_path, optimiser.struct, format='vasp')

        with open(os.path.join(out_dir, title), 'w') as f:
            f.write(optimiser.struct.info['initial_spacegroup'] + '   ' + optimiser.struct.info['final_spacegroup'])

        final_force = np.linalg.norm(optimiser.struct.get_forces(), axis=1).max()
        np.save(os.path.join(out_dir, 'final.npy'), np.array([energy / len(atoms), final_force]))

        if final_force > FMAX:
            print('WARNING: Optimisation not converged')
            not_converged.append(name)

        os.chdir(DATA_DIR)

    print(f'Following systems did not converge: {not_converged}')
    print(f'Following systems changed despite using ase constraint: {changed_despite_constraint}')
    print('FINISHED')
