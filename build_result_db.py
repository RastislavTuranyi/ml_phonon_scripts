import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')
DATA_DIR = os.path.join(HOME_DIR, 'data')
INS_DIR = os.path.join(DATA_DIR, 'ins')
OPTIMISED_DIR = os.path.join(DATA_DIR, 'optimised')

KNOWN_MODELS = ['mace_mp', 'mace_off', 'chgnet', 'm3gnet', 'sevennet']


def parse_csv_data() -> dict[str, dict[str, str]]:
    """
    Parses a CSV file called `data.csv` in the INS_DIR which contains a mapping between structure
    file name, and both the deuteration and the instrument that the experimental data was recorded
    on.

    This method was deemed to be the easiest way of encoding the mapping between file name and
    instrument it was recorded on.

    Example file:
    ```
    Name   Instrument   Deuteration   FileName
    Acenapthene   TFXA   ''   acenapthene_48932
    Acetinilide   TFXA   ''   acetinilide_170702
    Acetinilide-D3   TFXA   D3   acetinilide_170702
    Acetinilide-D5   TFXA   D5   acetinilide_170702
    Acetinilide-D8   TFXA   D8   acetinilide_170702
    "Acetic acid OH"   TOSCA   ''   acetic_acid_8722
    "Acetic acid OD"   TOSCA   D   acetic_acid_8722
    ```

    Example output of this function:
    ```
    {
        'acenapthene_48932: {
            '': 'TFXA',
        },
        'acetinilide_170702': {
            '': 'TFXA',
            'D3': 'TFXA',
            'D5': 'TFXA',
            'D8: 'TFXA',
        },
        'acetic_acid_8722': {
            '': 'TOSCA',
            'D': 'TOSCA',
        },
    }
    ```

    :return: Dictionary with the mapping
    """
    result = {}

    with open(os.path.join(INS_DIR, 'data.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for line in reader:
            file_field = line[3]
            if not file_field or line[9] == 'data inaccessible':
                continue

            if ',' in file_field:
                for file in file_field.split(','):
                    key = file.strip().replace('.cif', '')
                    deuteration = line[2].lower()
                    try:
                        result[key]
                    except KeyError:
                        result[key] = {}

                    result[key][deuteration] = (line[1], line[4], line[5])
            else:
                key = file_field.strip().replace('.cif', '')
                deuteration = line[2].lower()
                try:
                    result[key]
                except KeyError:
                    result[key] = {}

                result[key][deuteration] = (line[1], line[4], line[5])

    return result


def get_specific_dir(base_dir: str, arch: str, model_path: str, cell: bool) -> tuple[str, str]:
    if os.path.exists(model_path):
        p = os.path.split(model_path)[-1]
        name = '_'.join([arch, p])
        results_dir = os.path.join(base_dir, name)
    else:
        name = '_'.join([arch, model_path])
        results_dir = os.path.join(base_dir, name)

    if cell:
        return os.path.join(results_dir, 'cell'), name + '_cell'
    else:
        return os.path.join(results_dir, 'no_cell'), name + '_no_cell'


def get_id(name: str) -> int | None:
    for i, val in enumerate(reversed(name)):
        try:
            int(val)
        except ValueError:
            break
    else:
        return None

    return int(name[len(name)-i:])


def get_analyse_result(path: str) -> str | None:
    possibilities = ['ACCEPTABLE', 'FAILED', 'WEIRD-OK', 'WEIRD-FAIL', 'OK', 'GREAT']

    for possibility in possibilities:
        if os.path.exists(os.path.join(path, possibility)):
            return possibility

    return None


def get_results(path: str):
    try:
        supercell = ' '.join(np.load(os.path.join(path, 'supercell.npy')))
    except FileNotFoundError:
        return None, None

    imaginary = get_analyse_result(path)

    return supercell, imaginary


def subselect_items(data: dict):
    for deuteration, values in data.items():
        if not deuteration or deuteration.isnumeric():
            yield deuteration, values


def main(args):
    data = parse_csv_data()
    if args.arch and args.model_path:
        create_one_db(data, args.arch, args.model_path, args.cell)
        return

    runs = glob.glob(os.path.join(OPTIMISED_DIR, '*', ''))
    for run in runs:
        name = os.path.split(run)[-1]
        if 'old' in name or 'fmax' in name:
            continue

        for arch in KNOWN_MODELS:
            if arch in name:
                break
        else:
            print(f'Could not gues model arch from directory name for "{name}"')
            continue

        model_path = name.replace(arch, '')[1:]
        cell_runs = glob.glob(os.path.join(run, '*', ''))
        for cell_run in cell_runs:
            cell = os.path.split(cell_run)[-1]
            create_one_db(data, arch, model_path, cell)


def create_one_db(data, arch, model_path, cell):
    results_dir, result_name = get_specific_dir(RESULTS_DIR, arch, model_path, cell)
    optimised_dir, _ = get_specific_dir(OPTIMISED_DIR, arch, model_path, cell)

    result = ['compound', 'id', 'instrument', 'method', 'temperature', 'optimisation',
              'supercell', 'imaginary_modes', 'subjective']
    for compound, value in data.items():
        for _, (instrument, method, temperature) in subselect_items(value):
            opt = get_optimisation(compound, optimised_dir)
            supercell, imaginary = get_results(os.path.join(results_dir, compound))

            result.append([compound, get_id(compound), instrument, method, temperature, opt,
                           supercell, imaginary, None])

    with open(os.path.join(RESULTS_DIR, result_name + '.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(result)


def get_optimisation(compound, optimised_dir):
    vasp = compound + '.vasp'
    if os.path.exists(os.path.join(optimised_dir, vasp)):
        opt = 'success'
    elif os.path.exists(os.path.join(optimised_dir, 'not_converged', vasp)):
        opt = 'not_converged'
    elif os.path.exists(os.path.join(optimised_dir, 'spacegroup_changed', vasp)):
        opt = 'spacegroup_changed'
    else:
        opt = None
    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for comparing computed INS spectra with experimental equivalents. '
                    'Takes the phonopy force constants and runs them through AbINS to calculate '
                    'the predicted INS spectrum. This is then plotted on the same plot as the '
                    'experimental results.'
    )
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='',
                        help='The "--model-path" parameter for Janus.')
    args = parser.parse_args()

    main(args)