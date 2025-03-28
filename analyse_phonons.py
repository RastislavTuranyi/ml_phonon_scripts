import argparse
from contextlib import redirect_stdout
import glob
from multiprocessing import Process
import os

import numpy as np

from euphonic import ForceConstants
from euphonic.util import mp_grid

from band_plot import plot_bands

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')

GRID = mp_grid((5, 5, 5))
IMAGINARY_MODE_TOLERANCE = 1e-3

EXIT_SUCCESS = 0
EXIT_NO_FILE = 123
EXIT_RUNTIME_ERROR = 455


def print_result(p, img, imgc, result, phonons_correction):
    print(f'{result}: {np.sum(img)} imaginary modes, {np.sum(imgc)} with correction')
    print(f'og: {np.min(p, axis=0)[img]}')
    print(f'with correction: {np.min(phonons_correction, axis=0)[imgc]}')


def write_default(name, path, p, img, pc, imgc):
    with open(os.path.join(path, name), 'w') as f:
        f.write(f'og: {np.min(p, axis=0)[img]}\n')
        f.write(f'with correction: {np.min(pc, axis=0)[imgc]}\n')


def write_weird(name, path, p, img, pc, imgc):
    with open(os.path.join(path, name), 'w') as f:
        f.write(f'og: {np.min(p, axis=0)[img]}\n')
        f.write(f'with correction: {np.min(pc, axis=0)[imgc]}\n')

        indices = np.where(imgc)[0]
        for idx in indices:
            f.write(f'{GRID[pc[:, idx] < 0]}\n')


def clear_previous_assessment(dir: str):
    for possible in ['ACCEPTABLE', 'FAILED', 'WEIRD-OK', 'WEIRD-FAIL', 'OK', 'GREAT']:
        path = os.path.join(dir, possible)
        if os.path.exists(path):
            os.remove(path)


def run_one(dir):
    compound = os.path.split(os.path.split(dir)[0])[-1]
    print(compound)

    phonopy_file = os.path.join(dir, f'{compound}-phonopy')
    try:
        os.symlink(phonopy_file + '.yml', phonopy_file + '.yaml')
    except FileExistsError:
        pass

    clear_previous_assessment(dir)

    out = os.path.join(dir, f'{compound}_frequencies.npy')
    out_correction = os.path.join(dir, f'{compound}_frequencies_corrected.npy')

    if not args.recompute and os.path.exists(out) and os.path.exists(out_correction):
        phonons = np.load(out)
        phonons_correction = np.load(out_correction)
    else:
        try:
            with redirect_stdout(None):
                force_constants = ForceConstants.from_phonopy(
                    path=dir,
                    summary_name=f'{compound}-phonopy.yaml',
                    fc_name=f'{compound}-force_constants.hdf5'
                )
        except RuntimeError:
            supercell = np.load(os.path.join(dir, 'supercell.npy'))
            print('euphonic failed - supercell=', supercell, ' det=',
                  np.linalg.det(supercell.reshape((3, 3))))
            print()
            exit(EXIT_RUNTIME_ERROR)
        except FileNotFoundError:
            print('No supercell\n')
            exit(EXIT_NO_FILE)

        phonons = force_constants.calculate_qpoint_phonon_modes(GRID).frequencies.magnitude
        phonons_correction = force_constants.calculate_qpoint_phonon_modes(GRID, asr='reciprocal')

        if args.plot:
            plot_bands(phonons_correction, str(os.path.join(dir, f'{compound}_bands.png')))

        phonons_correction = phonons_correction.frequencies.magnitude

        np.save(out, phonons)
        np.save(out_correction, phonons_correction)

    imaginary = np.sum(phonons < 0, axis=0) > 0
    imaginary_correction = np.sum(phonons_correction < 0, axis=0) > 0

    ia = np.any(imaginary)
    ica = np.any(imaginary_correction)

    if ia and ica:
        if np.all(np.abs(phonons_correction[phonons_correction < 0]) < args.tolerance):
            print_result(phonons, imaginary, imaginary_correction, 'ACCEPTABLE', phonons_correction)
            write_default('ACCEPTABLE', dir, phonons, imaginary, phonons_correction,
                          imaginary_correction)
        else:
            print_result(phonons, imaginary, imaginary_correction, 'FAILED', phonons_correction)
            write_default('FAILED', dir, phonons, imaginary, phonons_correction,
                          imaginary_correction)
    elif ica:
        if np.all(np.abs(phonons_correction[phonons_correction < 0]) < args.tolerance):
            print_result(phonons, imaginary, imaginary_correction, 'WEIRD-OK', phonons_correction)
            write_weird('WEIRD-OK', dir, phonons, imaginary, phonons_correction,
                        imaginary_correction)
        else:
            print_result(phonons, imaginary, imaginary_correction, 'WEIRD-FAIL', phonons_correction)
            write_weird('WEIRD-FAIL', dir, phonons, imaginary, phonons_correction,
                        imaginary_correction)
    elif ia:
        print_result(phonons, imaginary, imaginary_correction, 'OK', phonons_correction)
        write_default('OK', dir, phonons, imaginary, phonons_correction, imaginary_correction)
    else:
        print(f'GREAT!!! {np.sum(imaginary)} imaginary modes, {np.sum(imaginary_correction)} with correction')
        with open(os.path.join(dir, 'GREAT', 'w')) as f:
            pass

    print()
    exit(EXIT_SUCCESS)


def main(args):
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

    failed_supercells = []
    successful_supercells = []
    for dir in directories:
        process = Process(target=run_one, args=(dir,))
        process.start()
        process.join()

        if process.exitcode == EXIT_SUCCESS:
            successful_supercells.append(np.load(os.path.join(dir, 'supercell.npy')))
        elif process.exitcode == EXIT_RUNTIME_ERROR:
            failed_supercells.append(np.load(os.path.join(dir, 'supercell.npy')))
        elif process.exitcode != EXIT_NO_FILE:
            print('unexpected failure - possible segfault')

    np.save(os.path.join(results_dir, 'failed_supercells.npy'), failed_supercells)
    np.save(os.path.join(results_dir, 'successful_supercells.npy'), successful_supercells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Computes the band structure and plots the result')
    parser.add_argument('-r', '--recompute', action='store_true',
                        help='Recomputes phonons')
    parser.add_argument('-t', '--tolerance', type=float, default=IMAGINARY_MODE_TOLERANCE,
                        help='The tolerance for imaginary modes')
    args = parser.parse_args()

    main(args)
