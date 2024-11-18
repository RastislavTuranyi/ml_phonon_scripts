import argparse
import os

import numpy as np

import euphonic
from euphonic.plot import plot_1d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='The input path')
    parser.add_argument('-o', '--output', type=str,
                        help='The output path')
    args = parser.parse_args()

    path, name = os.path.split(args.input)

    phonons = euphonic.QpointPhononModes.from_phonopy(path, name)
    phonons.reorder_frequencies()

    bands = phonons.get_dispersion().split(btol=5.)
    y_max = 0

    for band in bands:
        band._y_data = band.y_data.to('reciprocal_centimeter').magnitude
        m = np.max(band._y_data)
        if m > y_max:
            y_max = m

    if y_max > 3000:
        y_max = np.ceil(y_max / 500) * 500
    else:
        y_max = np.ceil(y_max / 200) * 200
    #bands = bands.split(btol=5.)

    fig = plot_1d(bands, ylabel='Energy (meV)', color='#E94D36', alpha=0.8)

    for ax in fig.axes:
        ax.plot(ax.get_xlim(), [0, 0], alpha=0.5, c='#1E5DF8', linestyle=':', linewidth=1)
        ax.tick_params(labelsize=13)
        ax.set_ylim(top=y_max)

    fig.axes[-1].set_ylabel('Frequency ($cm^{-1}$)', fontsize=20, labelpad=16)
    
    fig.tight_layout()

    fig.savefig(args.output, dpi=2000)
