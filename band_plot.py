"""
Script for producing a customised phonopy-style band plot.
"""
import argparse
import os

import numpy as np

import euphonic
from euphonic.plot import plot_1d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for producing a customised phonopy-style '
                                                 'band plot.')
    parser.add_argument('-i', '--input', type=str,
                        help='The path to a phonopy bands.yaml file to plot.')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='The path to the picture to save. If only a file name is provided, '
                             'a file with the name is saved to the same directory as the input '
                             'file. If nothing is provided, the picture is saved to '
                             'input_path/{name}_bands.png')
    parser.add_argument('-u', '--units', choices=['energy', 'frequency', 'f', 'e'],
                        help='The units to use for plotting the y-axis. If "energy" or "e" is '
                             'specified, the y-axis will be in meV, and if "frequency" or "f" is '
                             'specified, the y-axis will be in 1/cm.')
    args = parser.parse_args()

    if args.units in ['energy', 'e']:
        unit, ylabel = 'meV', 'Energy (meV)'
    elif args.units in ['frequency', 'f']:
        unit, ylabel = 'reciprocal_centimeter', 'Frequency ($cm^{-1}$)'
    else:
        raise NotImplementedError()

    path, name = os.path.split(args.input)

    phonons = euphonic.QpointPhononModes.from_phonopy(path, name)
    phonons.reorder_frequencies()

    bands = phonons.get_dispersion().split(btol=5.)
    y_max = 0

    for band in bands:
        band._y_data = band.y_data.to(unit).magnitude
        m = np.max(band._y_data)
        if m > y_max:
            y_max = m

    if y_max > 3000:
        y_max = np.ceil(y_max / 500) * 500
    else:
        y_max = np.ceil(y_max / 200) * 200
    #bands = bands.split(btol=5.)

    fig = plot_1d(bands, ylabel=ylabel, color='#E94D36', alpha=0.8)

    for ax in fig.axes:
        ax.plot(ax.get_xlim(), [0, 0], alpha=0.5, c='#1E5DF8', linestyle=':', linewidth=1)
        ax.tick_params(labelsize=13)
        ax.set_ylim(top=y_max)

    fig.axes[-1].set_ylabel(ylabel, fontsize=20, labelpad=16)
    
    fig.tight_layout()

    if args.output:
        output = os.path.split(args.output)
        if output[0]:
            fig.savefig(args.output, dpi=2000)
        else:
            fig.savefig(str(os.path.join(path, args.output)), dpi=2000)
    else:
        out_name = name.split('.')[0].replace('-auto_bands', '_bands.png')
        fig.savefig(str(os.path.join(path, out_name)), dpi=2000)
