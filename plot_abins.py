"""
Script for the final stage of the workflow: computing the INS spectra from the computed phonons and
plotting those against the experimental spectra.

Assumptions:

1. The `run_phonon.py` script has been successfully run, and the results are present as outputted
   by that script
2. The `analyse_phonons.py` script has been run to check on the outputs of the phonon calculations,
   and the marker files are present in the output directories.
3. The experimental data, downloaded from the ISIS INS database, is present in `./data/ins` and
   named following a similar naming convention as the structure files have been.
4. A CSV file `data.csv` is present in `./data/ins`, which contains the mapping between structure
   file names and the instrument used in the neutron experiment. More details can be found in
   :py:func:`parse_csv_data`.
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from mantid.simpleapi import Abins

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HOME_DIR, 'results')
INS_DIR = os.path.join(HOME_DIR, 'data', 'ins')


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
            if not file_field:
                continue

            if ',' in file_field:
                for file in file_field.split(','):
                    key = file.strip().replace('.cif', '')
                    deuteration = line[2].lower()
                    try:
                        result[key]
                    except KeyError:
                        result[key] = {}

                    result[key][deuteration] = line[1]
            else:
                key = file_field.strip().replace('.cif', '')
                deuteration = line[2].lower()
                try:
                    result[key]
                except KeyError:
                    result[key] = {}

                result[key][deuteration] = line[1]

    return result


def parse_data_file(path: str) -> np.ndarray:
    """
    Parses a data (ASCII) file from the ISIS INS database
    (http://wwwisis2.isis.rl.ac.uk/INSdatabase/Theindex.asp). The file is assumed to be an output
    from Mantid.

    :param path: Path to the file to parse
    :return: The table of data from the file.
    """
    out = []
    delimiter = None
    with open(path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) > 1:
                if has_data_started(values):
                    break
            else:
                values = line.strip().split(',')
                if len(values) > 1 and has_data_started(values):
                    delimiter = ','
                    break
        else:
            raise Exception('parsing error')

        out.append([float(val.strip()) for val in line.strip().split(delimiter)])

        for line in f:
            try:
                out.append([float(val.strip()) for val in line.split(delimiter)])
            except ValueError:
                print(line, delimiter, line.split(delimiter))
                raise

    data = split_parsed_data(out)
    try:
        return np.array(data)
    except ValueError:
        for val in data:
            print(val)
        raise


def has_data_started(line: list[str]) -> bool:
    """
    Checks whether the data has started, as indicated by the fact that the line contains a row of
    float data, at least two values long.

    :param line: A split line of the data file
    :return: Whether the data has started
    """
    for item in line:
        try:
            float(item.strip())
        except ValueError:
            return False

    return True


def normalise_data(abins_x: np.ndarray,
                   abins_y: np.ndarray,
                   experimental: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Normalises the experimental data w.r.t. the computed data in order to make them appear to have
    similar scales on a plot. This is done by setting the highest peak above 50 $cm^{-1}$ in both
    datasets to be equal. The restriction is used because experimental data can contain a massive
    elastic peak near 0 $cm^{-1}$, which can mess with the normalisation.

    :param abins_x: The computed frequency data
    :param abins_y: The computed S(q, w)
    :param experimental: The experimental data in a 2D array, where the columns are the x and y data

    :return: The normalised data, and the intensity of the highest peak above 50 $cm^{-1}$
    """
    abins_start = np.where(abins_x > 50)[0][0]
    exp_start = np.where(experimental[:, 0] > 50)[0][0]

    abins_max = np.max(abins_y[abins_start:])
    exp_max = np.max(experimental[exp_start:, 1])

    experimental[:, 1] *= abins_max / exp_max

    return abins_x, abins_y, experimental, max([abins_max, exp_max])


def split_parsed_data(data: list[list[float]]) -> list[list[float]]:
    """
    Some Mantid outputs contain multiple tables of data in one file, corresponding to the partial
    and total S(q, w). This function discards all but the last one, which is assumed to be the total
    S(q, w). The tables are assumed to be separated by a single value (as opposed to two or three
    columns of the data itself.

    :param data: The contents of the data file in a list format.

    :return: The last table, marked `2`, in the file.
    """
    out = []

    for i, line in enumerate(data):
        if len(line) == 1 and int(line[0]) == 2:
            break
    else:
        return data

    for line in data[i+1:]:
        out.append(line)

    return out


def subselect_items(data: dict, force_tosca: bool = False):
    result = []
    for deuteration, instrument in data.items():
        if not deuteration or deuteration.isnumeric():
            if force_tosca:
                if instrument != '?':
                    result.append((deuteration, instrument))
            else:
                if instrument.lower() == 'tosca':
                    result.append((deuteration, instrument))

    return result


def main(args):
    data = parse_csv_data()

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

    for directory in directories:
        compound = os.path.split(os.path.split(directory)[0])[-1]

        if not args.replot and os.path.exists(os.path.join(directory, f'{compound}.png')):
            print(f'Skipping {compound} because already complete')
            continue

        print()
        if not (
                os.path.exists(os.path.join(directory, 'ACCEPTABLE')) or
                os.path.exists(os.path.join(directory, 'WEIRD-OK')) or
                os.path.exists(os.path.join(directory, 'OK')) or
                os.path.exists(os.path.join(directory, 'GREAT'))
        ):
            print(f'skipping {compound} because of imaginary modes')
            continue

        non_deuterated = subselect_items(data[compound], args.force_tosca)
        if not non_deuterated:
            print(f'skipping {compound} due to not having TOSCA measurements')
            continue

        try:
            os.symlink(os.path.join(directory, f'{compound}-force_constants.hdf5'),
                       os.path.join(directory, 'force_constants.hdf5'))
        except FileExistsError:
            pass

        print(compound)

        energy, result, s = get_abins_data(compound, directory)

        for deuteration, instrument in non_deuterated:
            name = f'{compound}_{deuteration}.dat' if deuteration else f'{compound}.dat'
            ins_data = parse_data_file(os.path.join(INS_DIR, name))

            energy, s, ins_data, y_max = normalise_data(energy, s, ins_data)

            plot_abins(name[:-4], directory, energy, ins_data, s, y_max)

        try:
            result.delete()
        except (NameError, AttributeError):
            pass

    created_hdf_files = glob.glob(os.path.join(HOME_DIR, '*.hdf5'))
    for file in created_hdf_files:
        os.remove(file)


def get_abins_data(compound, directory):
    if os.path.exists(os.path.join(directory, f'abins.npy')):
        result = np.load(os.path.join(directory, 'abins.npy'))
        energy = result[0, :]
        s = result[1, :]
    else:
        result = Abins(
            VibrationalOrPhononFile=os.path.join(directory, f'{compound}-phonopy.yaml'),
            AbInitioProgram='FORCECONSTANTS',
            Instrument='TOSCA',
            SumContributions=True,
            QuantumOrderEventsNumber='2',
            Autoconvolution=True,
            Setting='All detectors (TOSCA)',
            ScaleByCrossSection="Total")

        energy = result[0].extractX().flatten()
        energy = (energy[1:] + energy[:-1]) / 2
        s = result[0].extractY().flatten()

        np.save(os.path.join(directory, 'abins.npy'),
                np.stack([energy, s]))

    return energy, result, s


def plot_abins(compound, directory, energy, ins_data, s, y_max):
    fig, ax = plt.subplots(dpi=2000)

    ax.plot(ins_data[:, 0], ins_data[:, 1], label='Experimental', alpha=0.7, c='#1E5DF8',
            linewidth=2.5)
    ax.plot(energy, s, label='AbINS', alpha=0.7, c='#E94D36', linewidth=2.5)

    ax.set_xlabel('Energy transfer $(cm^{-1})$', fontsize=20)
    ax.set_ylabel('S(q, w)', fontsize=20)

    ax.set_xlim(0, 4000)
    if np.max(ins_data[:, 1]) > 3 * y_max:
        y_min = min([np.min(s), np.min(ins_data[:, 1])])
        ax.set_ylim(y_min * 0.9, y_max * 1.5)

    ax.tick_params(length=5, width=2, labelsize=15)
    ax.axes.get_yaxis().set_ticks([])

    plt.legend(fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(directory, f'{compound}.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for comparing computed INS spectra with experimental equivalents. '
                    'Takes the phonopy force constants and runs them through AbINS to calculate '
                    'the predicted INS spectrum. This is then plotted on the same plot as the '
                    'experimental results.'
    )
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-rp', '--replot', action='store_true',
                        help='Disables skipping when the plot already exists.')
    parser.add_argument('-ft', '--force-tosca', action='store_true',
                        help='Forces the TOSCA resolution to be used for all compounds, regardless '
                             'of which instrument they were measured on.')
    args = parser.parse_args()

    main(args)
