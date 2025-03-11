from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from scipy.signal import butter, filtfilt

from plot_abins import parse_data_file


HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')
DATA_DIR = os.path.join(HOME_DIR, 'data')
INS_DIR = os.path.join(DATA_DIR, 'ins')
OPTIMISED_DIR = os.path.join(DATA_DIR, 'optimised')

KNOWN_MODELS = ['mace_mp', 'mace_off', 'chgnet', 'm3gnet', 'sevennet']

DATA_COMPARISON_LOWER_BOUND = 50  # cm^{-1}


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

            line[4] = line[4] if line[4] else '?'
            line[5] = line[5] if line[5] else '?'

            if ', ' in file_field:
                for file in file_field.split(', '):
                    key = file.strip().replace('.cif', '')
                    deuteration = line[2].lower()
                    try:
                        result[key]
                    except KeyError:
                        result[key] = {}

                    result[key][deuteration] = (line[1], line[4], line[5], line[12])
            else:
                key = file_field.strip().replace('.cif', '')
                deuteration = line[2].lower()
                try:
                    result[key]
                except KeyError:
                    result[key] = {}

                result[key][deuteration] = (line[1], line[4], line[5], line[12])

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

    try:
        return int(name[len(name)-i:])
    except ValueError:
        print(name, i)
        raise

def get_analyse_result(path: str) -> str | None:
    possibilities = ['ACCEPTABLE', 'FAILED', 'WEIRD-OK', 'WEIRD-FAIL', 'OK', 'GREAT']

    for possibility in possibilities:
        if os.path.exists(os.path.join(path, possibility)):
            return possibility

    return None


def get_results(path: str):
    try:
        supercell = ' '.join(np.load(os.path.join(path, 'supercell.npy')).astype(str))
    except FileNotFoundError:
        return None, None

    imaginary = get_analyse_result(path)

    return supercell, imaginary


def subselect_items(data: dict):
    for deuteration, values in data.items():
        if not deuteration or deuteration.isnumeric():
            yield deuteration, values


def main(args):
    if args.arch and args.model_path:
        if args.update:
            update_one_db(args.arch, args.model_path, args.cell)
        else:
            data = parse_csv_data()
            create_one_db(data, args.arch, args.model_path, args.cell)
    else:
        if args.update:
            update_multiple_db()
        else:
            create_multiple_db()


def create_multiple_db():
    data = parse_csv_data()

    runs = glob.glob(os.path.join(OPTIMISED_DIR, '*', ''))
    for run in runs:
        name = os.path.split(os.path.dirname(run))[-1]
        if 'old' in name or 'fmax' in name:
            print(f'skipping {name}')
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
            cell = os.path.split(os.path.dirname(cell_run))[-1]
            print(f'{name}_{cell}')
            create_one_db(data, arch, model_path, cell)


def update_multiple_db():
    runs = glob.glob(os.path.join(RESULTS_DIR, '*.csv'))
    for run in runs:
        print(os.path.split(run)[-1])
        update_one_db(csv_path=run)


def create_one_db(data, arch, model_path, cell):
    results_dir, result_name = get_specific_dir(RESULTS_DIR, arch, model_path, cell)
    optimised_dir, _ = get_specific_dir(OPTIMISED_DIR, arch, model_path, cell)

    result = [['compound', 'id', 'instrument', 'method', 'temperature', 'optimisation',
              'supercell', 'imaginary_modes', 'score_filtered', 'score_direct', 'which',
               'is_organic', 'is_inorganic', 'is_organometallic', 'is_polymeric', 'formula',
               'n_rings', 'n_aromatic_rings', 'n_fused_rings']]
    for compound, value in data.items():
        compound_result_dir = os.path.join(results_dir, compound)
        try:
            abins_data = np.load(os.path.join(compound_result_dir, 'abins.npy'))
        except FileNotFoundError:
            abins_data = None

        for deuteration, (instrument, method, temperature, id) in subselect_items(value):
            opt = get_optimisation(compound, optimised_dir)
            supercell, imaginary = get_results(compound_result_dir)

            if abins_data is None:
                score1, score2 = None, None
            else:
                name = f'{compound}_{deuteration}.dat' if deuteration else f'{compound}.dat'
                ins_data = parse_data_file(os.path.join(INS_DIR, name))
                score1 = compare_abins_ins_filtered(abins_data, ins_data)
                score2 = compare_abins_ins_direct(abins_data, ins_data)

            if not id:
                id = get_id(compound)

            result.append([compound, id, instrument, method.lower(), temperature, opt,
                           supercell, imaginary, score1, score2] + [None] * 9)

    with open(os.path.join(RESULTS_DIR, result_name + '.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(result)

    with open(os.path.join(RESULTS_DIR, result_name + '.csv.tag'), 'w') as f:
        f.write(f'{arch}\n{model_path}\n{cell}\n')


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


def compare_abins_ins_filtered(abins_data, ins_data, lower_bound=DATA_COMPARISON_LOWER_BOUND):
    ins_x, ins_y = ins_data[:, 0], ins_data[:, 1]
    abins_x, abins_y = abins_data[0, :], abins_data[1, :]

    keep_idx = ins_x >= lower_bound
    ins_x, ins_y = ins_x[keep_idx], ins_y[keep_idx]

    if ins_x[0] > abins_x[0]:
        keep_idx = abins_x >= ins_x[0]
        abins_x, abins_y = abins_x[keep_idx], abins_y[keep_idx]

    if ins_x[-1] < abins_x[-1]:
        keep_idx = abins_x <= ins_x[-1]
        abins_x, abins_y = abins_x[keep_idx], abins_y[keep_idx]

    ins_y_interpolated = interp1d(ins_x, ins_y, kind='cubic')(abins_x)
    b, a = butter(2, 1e-3, btype='high')

    ins_y_filtered = filtfilt(b, a, ins_y_interpolated)
    abins_y_filtered = filtfilt(b, a, abins_y)

    ins_y_filtered /= np.trapz(np.abs(ins_y_filtered), ins_x)
    abins_y_filtered /= np.trapz(np.abs(abins_y_filtered), abins_x)

    return wasserstein_distance(ins_y_filtered, abins_y_filtered) * (np.max(abins_x) - np.min(abins_x))


def compare_abins_ins_direct(abins_data, ins_data, lower_bound=DATA_COMPARISON_LOWER_BOUND):
    ins_x, ins_y = ins_data[:, 0], ins_data[:, 1]
    abins_x, abins_y = abins_data[0, :], abins_data[1, :]

    keep_idx = ins_x >= lower_bound
    ins_x, ins_y = ins_x[keep_idx], ins_y[keep_idx]

    if ins_x[0] > abins_x[0]:
        keep_idx = abins_x >= ins_x[0]
        abins_x, abins_y = abins_x[keep_idx], abins_y[keep_idx]

    if ins_x[-1] < abins_x[-1]:
        keep_idx = abins_x <= ins_x[-1]
        abins_x, abins_y = abins_x[keep_idx], abins_y[keep_idx]

    ins_y /= np.trapz(ins_y, ins_x)
    abins_y /= np.trapz(abins_y, abins_x)

    ins_y_interpolated = interp1d(ins_x, ins_y, kind='cubic')(abins_x)

    return wasserstein_distance(ins_y_interpolated, abins_y) * (np.max(abins_x) - np.min(abins_x))


def update_one_db(arch=None, model_path=None, cell=None, csv_path=None):
    from ccdc.search import TextNumericSearch
    from ccdc.entry import Entry

    if csv_path is not None:
        pass
    elif arch is not None and model_path is not None and cell is not None:
        _, result_name = get_specific_dir(RESULTS_DIR, arch, model_path, cell)
        csv_path = os.path.join(RESULTS_DIR, result_name + '.csv')
    else:
        raise Exception()

    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data.append(next(reader))
        for line in reader:
            if not line:
                continue
            query = TextNumericSearch()
            query.add_ccdc_number(int(line[1]))
            result = query.search()

            if result:
                result = result[0].entry
                line[10] = 'organometallic' if result.is_organometallic else 'organic'
                line[11] = result.is_organic
                line[12] = False
                line[13] = result.is_organometallic
            else:
                try:
                    with open(os.path.join(DATA_DIR, line[0] + '.cif'), 'r') as cif:
                        result = Entry.from_string(''.join(cif.readlines()))
                except FileNotFoundError:
                    with open(os.path.join(HOME_DIR, 'difficult', line[0] + '.cif'), 'r') as cif:
                        result = Entry.from_string(''.join(cif.readlines()))

                line[10] = 'organometallic' if result.is_organometallic else 'inorganic'
                line[11] = False
                line[12] = True
                line[13] = result.is_organometallic

            line[14] = result.is_polymeric
            line[15] = result.molecule.formula
            line[16] = len(result.molecule.rings)
            line[17] = len([ring for ring in result.molecule.rings if ring.is_aromatic])
            line[18] = len([ring for ring in result.molecule.rings if ring.is_fused])

            data.append(line)

    with open(csv_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)


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
    parser.add_argument('-u', '--update', action='store_true',
                        help='Updates the csv file(s) with data from CSD (requires CSD Python API)')
    args = parser.parse_args()

    main(args)
