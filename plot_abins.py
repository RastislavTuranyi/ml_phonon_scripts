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


def parse_csv_data():
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


def parse_data_file(path):
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

    return split_parsed_data(out)


def has_data_started(line):
    for item in line:
        try:
            float(item.strip())
        except ValueError:
            return False

    return True


def normalise_data(abins_x, abins_y, experimental):
    #abins_y -= abins_y[-1]
    #experimental[:, 1] -= experimental[-1, 1]
    
    abins_start = np.where(abins_x > 50)[0][0]
    exp_start = np.where(experimental[:, 0] > 50)[0][0]

    abins_max = np.max(abins_y[abins_start:])
    exp_max = np.max(experimental[exp_start:, 1])

    experimental[:, 1] *= abins_max / exp_max

    return abins_x, abins_y, experimental, max([abins_max, exp_max])


def split_parsed_data(data: list[list[float]]):
    out = []

    for i, line in enumerate(data):
        if len(line) == 1 and int(line[0]) == 2:
            break
    else:
        return data

    for line in data[i+1:]:
        out.append(line)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
    parser.add_argument('-rp', '--replot', action='store_true',
                        help='Disables skipping when the plot already exists.')
    args = parser.parse_args()

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

        try:
            if data[compound][''].lower() != 'tosca':
                print(f'skipping {compound} due to not having TOSCA measurements')
                continue
        except KeyError:
            print(f'skipping {compound} due to not having TOSCA measurements')
            continue

        try:
            os.symlink(os.path.join(directory, f'{compound}-force_constants.hdf5'),
                       os.path.join(directory, 'force_constants.hdf5'))
        except FileExistsError:
            pass

        print(compound)

        if os.path.exists(os.path.join(directory, f'abins.npy')):
            result = np.load(os.path.join(directory, 'abins.npy'))
            energy = result[0, :]
            s = result[1, :]
        else:
            result = Abins(VibrationalOrPhononFile=os.path.join(directory, f'{compound}-phonopy.yaml'),
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

        ins_data = parse_data_file(os.path.join(INS_DIR, f'{compound}.dat'))
        try:
            ins_data = np.array(ins_data)
        except ValueError:
            for val in ins_data:
                print(val)
            raise

        energy, s, ins_data, y_max = normalise_data(energy, s, ins_data)

        fig, ax = plt.subplots(dpi=2000)

        ax.plot(ins_data[:, 0], ins_data[:, 1], label='Experimental', alpha=0.7, c='#1E5DF8', linewidth=2.5)
        ax.plot(energy, s, label='AbINS', alpha=0.7, c='#E94D36', linewidth=2.5)

        ax.set_xlabel('Energy transfer $(cm^{-1})$', fontsize=20)
        ax.set_ylabel('S(q, w)', fontsize=20)

        ax.set_xlim(0, 4000)
        
        if np.max(ins_data[:, 1]) > 3 * y_max:
            y_min = min([np.min(s), np.min(ins_data[:, 1])])
            ax.set_ylim(y_min*0.9, y_max*1.5)

        ax.tick_params(length=5, width=2, labelsize=15)
        ax.axes.get_yaxis().set_ticks([])

        plt.legend(fontsize=15)

        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'{compound}.png'))
        plt.close(fig)

        try:
            result.delete()
        except (NameError, AttributeError):
            pass

    created_hdf_files = glob.glob(os.path.join(HOME_DIR, '*.hdf5'))
    for file in created_hdf_files: 
        os.remove(file)
