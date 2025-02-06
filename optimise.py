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
MAX_STEPS = 2000
DEFAULT_FILTER_FUNC = 'FrechetCellFilter'


def recompute_changed(original_file: str,
                      original_dir: str,
                      original_name: str,
                      source_file: str,
                      cell: bool,
                      arch: str,
                      model_path: str,
                      filter_func,
                      fkwargs: dict,
                      okwargs: dict,
                      tkwargs: dict,
                      dispersion: bool = True,
                      fmax=FMAX):
    # Move output file to extra data dir and rename the dir
    os.rename(original_file, os.path.join(original_dir, original_name))
    try:
        os.rename(original_dir, original_dir + '_changed')
    except FileExistsError:
        rmtree(original_dir + '_changed')
        os.rename(original_dir, original_dir + '_changed')

    os.makedirs(original_dir)
    os.chdir(original_dir)

    atoms = read(source_file, format='vasp')
    atoms.set_constraint(FixSymmetry(atoms=atoms, adjust_positions=True, adjust_cell=cell))
    optimiser = GeomOpt(struct=atoms,
                        arch=arch,
                        device='cuda',
                        model_path=model_path,
                        calc_kwargs={'dispersion': dispersion},
                        attach_logger=True,
                        fmax=fmax,
                        steps=MAX_STEPS,
                        write_results=True,
                        filter_func=filter_func,
                        filter_kwargs=fkwargs,
                        opt_kwargs=okwargs,
                        traj_kwargs=tkwargs,
                        track_carbon=False)
    optimiser.run()
    final_force = np.linalg.norm(optimiser.struct.get_forces(), axis=1).max()

    same = optimiser.struct.info['initial_spacegroup'] == optimiser.struct.info['final_spacegroup']
    if same:
        title = 'spacegroup_conserved'
    else:
        print('Space group changed despite ASE constraint')
        title = 'spacegroup_changed'

    write(original_file, optimiser.struct, format='vasp')

    with open(os.path.join(original_dir, title), 'w') as f:
        f.write(optimiser.struct.info['initial_spacegroup'] + '   ' + optimiser.struct.info['final_spacegroup'])

    np.save(os.path.join(original_dir, 'final.npy'),
            np.array([optimiser.struct.get_potential_energy(), final_force]))

    return final_force, same


def check_cif2cell_vesta(save_dir: str, top_dir: str, arch: str, model_path: str, dispersion: bool = True):
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
            vesta_energy = compute_one_energy(vesta_file, arch, model_path, dispersion)

        cif2cell_dir = os.path.join(cif2cell_file.replace('.vasp', ''), 'extra_data', cif2cell_name)
        cif2cell_energy = os.path.join(cif2cell_dir, 'final.npy')
        if os.path.exists(cif2cell_energy):
            cif2cell_energy = np.load(cif2cell_energy)[0]
        else:
            cif2cell_energy = compute_one_energy(cif2cell_file, arch, model_path, dispersion)

        if cif2cell_energy > vesta_energy:
            print('VESTA file lower in energy')
            os.rename(cif2cell_file, os.path.join(duplicates_dir, cif2cell_name + '.vasp'))
        else:
            os.rename(vesta_file, os.path.join(duplicates_dir, vesta_name + '.vasp'))
            print('cif2cell file lower in energy')


def compute_one_energy(file_path: str, arch: str, model_path: str, dispersion: bool = True) -> float:
    atoms = read(file_path, format='vasp')
    sp = SinglePoint(struct=atoms,
                     arch=arch,
                     device='cuda',
                     model_path=model_path,
                     calc_kwargs={'dispersion': dispersion},
                     properties='energy',
                     track_carbon=False)

    result = sp.run()
    return result['energy'] / len(atoms)


def run_geometry_optimisation(atoms: ase.Atoms,
                              arch: str,
                              model_path: str,
                              filter_func,
                              filter_kwargs: dict,
                              okwargs: dict,
                              tkwargs: dict,
                              dispersion: bool = True,
                              fmax: float = FMAX):
    optimiser = GeomOpt(struct=atoms,
                        arch=arch,
                        device='cuda',
                        model_path=model_path,
                        calc_kwargs={'dispersion': dispersion},
                        attach_logger=True,
                        fmax=fmax,
                        steps=MAX_STEPS,
                        write_results=True,
                        filter_func=filter_func,
                        filter_kwargs=filter_kwargs,
                        opt_kwargs=okwargs,
                        traj_kwargs=tkwargs,
                        track_carbon=False)
    optimiser.run()
    return optimiser


def set_subdir(parent_dir: str, name: str) -> str:
    sub_dir = os.path.join(parent_dir, name)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-sf', '--skip-failed', action='store_true',
                        help='Causes previously known failed calculations (due to either symmetry having changed or '
                             'because it did not converge) to be skipped instead of recomputing.')
    parser.add_argument('-dd', '--disable-dispersion', action='store_true', help='Disables dispersion')
    parser.add_argument('-f', '--fmax', type=float, help=f'The FMAX to use for optimisation ({FMAX} by default)')
    args = parser.parse_args()

    dispersion = not args.disable_dispersion
    filter_func = DEFAULT_FILTER_FUNC if args.cell else None
    filter_kwargs = {'hydrostatic_strain': args.cell}

    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, p]))
    else:
        target_dir = os.path.join(TARGET_DIR, '_'.join([args.arch, args.model_path]))

    args.model_path = None if args.model_path == 'None' else args.model_path

    target_dir = os.path.join(target_dir, 'cell') if args.cell else os.path.join(target_dir, 'no_cell')

    if args.restart:
        rmtree(target_dir)
        os.makedirs(target_dir)
    
    extra_dir = set_subdir(target_dir, 'extra_data')
    not_converged_dir = set_subdir(target_dir, 'not_converged')
    sg_changed_dir = set_subdir(target_dir, 'spacegroup_changed')

    files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.vasp')))

    not_converged, changed_despite_constraint = [], []
    for file in files:
        name = os.path.split(file)[-1]
        print(name)

        out_path = os.path.join(target_dir, name)
        out_dir = os.path.join(extra_dir, name.replace('.vasp', ''))
        traj_kwargs = {'filename': os.path.join(out_dir, 'optimisation.traj')}
        opt_kwargs = {'trajectory': traj_kwargs['filename']}

        if os.path.exists(out_path):
            print(f'Skipping {name} because it is already complete')

            if os.path.exists(os.path.join(out_dir, 'spacegroup_changed')):
                print('Recomputing data because space group in the original was changed')
                final_force, sg_same = recompute_changed(out_path, out_dir, name, file, args.cell, args.arch,
                                                         args.model_path, filter_func, filter_kwargs, opt_kwargs, traj_kwargs,
                                                         dispersion, fmax=args.fmax)
                if final_force > args.fmax:
                    print('WARNING: Constrained optimisation did not converge')
                    not_converged.append(name)
                    os.rename(out_path, os.path.join(not_converged_dir, name))
                else:
                    if sg_same:
                        # Remove trajectory if everything ok
                        os.remove(traj_kwargs['filename'])
                    else:
                        os.rename(out_path, os.path.join(sg_changed_dir, name))
            continue
        elif os.path.exists(out_dir):
            if os.path.exists(os.path.join(target_dir, 'high_energy_structures', name)):
                print('Skipping because the structure is already complete and has been placed to high_energy_structures')
                continue
            elif (os.path.exists(os.path.join(not_converged_dir, name)) or
                  os.path.exists(os.path.join(sg_changed_dir, name))):
                if args.skip_failed:
                    print('Skipping because optimisation failed and skipping was requested')
                    continue
                else:
                    print('Optimisation failed previously; redoing from start')
            else:
                print('Previously, optimisiation was started but not finished; starting over')

            rmtree(out_dir)

        os.makedirs(out_dir)
        copyfile(file, os.path.join(out_dir, name))
        os.chdir(out_dir)

        atoms = read(file, format='vasp')
        optimiser = run_geometry_optimisation(atoms, args.arch, args.model_path, filter_func, filter_kwargs,
                                              opt_kwargs, traj_kwargs, dispersion, fmax=args.fmax)
        energy = optimiser.struct.get_potential_energy()

        sg_different = optimiser.struct.info['initial_spacegroup'] != optimiser.struct.info['final_spacegroup']
        if sg_different:
            print('Space group changed during optimisation -> retrying with fixed symmetry')

            atoms = read(file, format='vasp')
            atoms.set_constraint(FixSymmetry(atoms=atoms, adjust_positions=True, adjust_cell=args.cell))
            optimiser = run_geometry_optimisation(atoms, args.arch, args.model_path, filter_func, filter_kwargs,
                                                  opt_kwargs, traj_kwargs, dispersion, fmax=args.fmax)

            energy2 = optimiser.struct.get_potential_energy()
            print(f'Original energy: {energy}; new energy: {energy2}')
            energy = energy2

            sg_different = optimiser.struct.info['initial_spacegroup'] != optimiser.struct.info['final_spacegroup']
            if not sg_different:
                title = 'spacegroup_conserved'
            else:
                print('Space group changed despite ASE constraint')
                title = 'spacegroup_changed'
                changed_despite_constraint.append(name)
        else:
            print('space group not changed')
            title = 'spacegroup_conserved'
        
        final_force = np.linalg.norm(optimiser.struct.get_forces(), axis=1).max()
        np.save(os.path.join(out_dir, 'final.npy'), np.array([energy / len(atoms), final_force]))

        write(out_path, optimiser.struct, format='vasp')

        if final_force > args.fmax:
            print('WARNING: Optimisation not converged')
            not_converged.append(name)
            os.rename(out_path, os.path.join(not_converged_dir, name))
        else:
            if not sg_different:
                # Remove trajectory if everything ok
                os.remove(traj_kwargs['filename'])
            else:
                os.rename(out_path, os.path.join(sg_changed_dir, name))

        try:
            with open(os.path.join(out_dir, title), 'w') as f:
                f.write(optimiser.struct.info['initial_spacegroup'] + '   ' + optimiser.struct.info['final_spacegroup'])
        except TypeError:
            if final_force < args.fmax:
                raise

        os.chdir(DATA_DIR)

    print(f'Following systems did not converge: {not_converged}')
    print(f'Following systems changed despite using ase constraint: {changed_despite_constraint}')
    print('FINISHED')
