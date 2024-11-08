import argparse
import glob
import os

import numpy as np

from euphonic import ForceConstants
from euphonic.util import mp_grid

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')

GRID = mp_grid((5, 5, 5))
IMAGINARY_MODE_TOLERANCE = 1e-3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-t', '--tolerance', type=float, default=IMAGINARY_MODE_TOLERANCE,
                        help='The tolerance for imaginary modes')
    args = parser.parse_args()


    if os.path.exists(args.model_path):
        p = os.path.split(args.model_path)[-1]
        results_dir = os.path.join(RESULTS_DIR, '_'.join([args.arch, p]))
    else:
        results_dir = os.path.join(RESULTS_DIR, '_'.join([args.arch, args.model_path]))

    if args.cell:
        results_dir = os.path.join(results_dir, 'cell')
    else:
        results_dir = os.path.join(results_dir, 'no_cell')

    directories = glob.glob(os.path.join(results_dir, '*', ''))

    for dir in directories:
        compound = os.path.split(os.path.split(dir)[0])[-1]
        print(compound)

        phonopy_file = os.path.join(dir, f'{compound}-phonopy')
        try:
            os.symlink(phonopy_file + '.yml', phonopy_file + '.yaml')
        except FileExistsError:
            pass

        out = os.path.join(dir, f'{compound}_frequencies.npy')
        out_correction = os.path.join(dir, f'{compound}_frequencies_corrected.npy')

        if os.path.exists(out) and os.path.exists(out_correction):
            phonons = np.load(out)
            phonons_correction = np.load(out_correction)
        else:
            force_constants = ForceConstants.from_phonopy(
                path=dir,
                summary_name=f'{compound}-phonopy.yaml',
                fc_name=f'{compound}-force_constants.hdf5'
            )

            phonons = force_constants.calculate_qpoint_phonon_modes(GRID).frequencies.magnitude
            phonons_correction = force_constants.calculate_qpoint_phonon_modes(GRID, asr='reciprocal').frequencies.magnitude

            np.save(out, phonons)
            np.save(out_correction, phonons_correction)

        imaginary = np.sum(phonons < 0, axis=0) > 0
        imaginary_correction = np.sum(phonons_correction < 0, axis=0) > 0

        ia = np.any(imaginary)
        ica = np.any(imaginary_correction)

        if ia and ica:
            if np.all(np.abs(phonons_correction[phonons_correction < 0]) < args.tolerance):
                print(f'ACCEPTABLE: {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
                print(f'og: {np.min(phonons, axis=0)[imaginary]}')
                print(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}')

                with open(os.path.join(dir, 'ACCEPTABLE'), 'w') as f:
                    f.write(f'og: {np.min(phonons, axis=0)[imaginary]}\n')
                    f.write(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}\n')
            else:
                print(f'FAILED: {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
                print(f'og: {np.min(phonons, axis=0)[imaginary]}')
                print(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}')

                with open(os.path.join(dir, 'FAILED'), 'w') as f:
                    f.write(f'og: {np.min(phonons, axis=0)[imaginary]}\n')
                    f.write(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}\n')
        elif ica:
            print(f'WEIRD: {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
            print(f'og: {np.min(phonons, axis=0)[imaginary]}')
            print(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}')

            with open(os.path.join(dir, 'WEIRD'), 'w') as f:
                f.write(f'og: {np.min(phonons, axis=0)[imaginary]}\n')
                f.write(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}\n')

                indices = np.where(imaginary_correction)[0]
                for idx in indices:
                    f.write(f'{GRID[phonons_correction[:, idx] < 0]}\n')
        elif ia:
            print(f'OK: {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
            print(f'og: {np.min(phonons, axis=0)[imaginary]}')
            print(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}')

            with open(os.path.join(dir, 'OK'), 'w') as f:
                f.write(f'og: {np.min(phonons, axis=0)[imaginary]}\n')
                f.write(f'with correction: {np.min(phonons_correction, axis=0)[imaginary_correction]}\n')
        else:
            print(f'GREAT!!! {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
            with open(os.path.join(dir, 'GREAT', 'w')) as f:
                pass

        print()
