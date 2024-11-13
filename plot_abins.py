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
            line = line.strip().split()
            if len(line) > 1:
                if has_data_started(line):
                    break
            else:
                line = line.strip().split(',')
                if len(line) > 1 and has_data_started(line):
                    break
        else:
            raise Exception('parsing error')

        out.append([float(val.strip()) for val in line])

        for line in f:
            out.append([float(val.strip()) for val in line])

    return out


def has_data_started(line):
    for item in line:
        try:
            float(item.strip())
        except TypeError:
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true',
                        help='If provided, the cell parameters are optimised')
    parser.add_argument('-a', '--arch', type=str, default='mace_mp',
                        help='The "--arch" parameter for Janus.')
    parser.add_argument('-mp', '--model-path', type=str, default='large',
                        help='The "--model-path" parameter for Janus.')
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
        print()
        compound = os.path.split(os.path.split(directory)[0])[-1]

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

        result = Abins(VibrationalOrPhononFile=os.path.join(directory, f'{compound}-phonopy.yaml'),
                       AbInitioProgram='FORCECONSTANTS',
                       Instrument='TOSCA',
                       SumContributions=True)

        energy = result[0].extractX().flatten()
        s = result[0].extractY().flatten()
        print(np.shape(energy), np.shape(s))
        np.save(os.path.join(directory, 'abins.npy'),
                np.stack([energy, s]))

        ins_data = np.array(parse_data_file(os.path.join(INS_DIR, f'{compound}.dat')))

        fig, ax = plt.subplots(dpi=400)

        ax.plot(ins_data[:, 0], ins_data[:, 1], label='Experimental')
        ax.plot(energy, s, label='AbINS')

        ax.set_xlabel('Energy transfer $(cm^{-1})$')
        ax.set_ylabel('S (unknown units)')

        plt.legend()

        fig.savefig(os.path.join(directory, f'{compound}.png'))
        plt.close(fig)
