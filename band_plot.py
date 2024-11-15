import argparse
import os

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

    bands = phonons.get_dispersion()
    for band in bands:
        band.y_data = band.y_data.to('reciprocal_centimeter')

    bands = bands.split(btol=5.)

    fig = plot_1d(bands, ylabel='Energy (meV)', color='#E94D36', alpha=0.8)

    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_color('#E94D36')

    fig.axes[-1].set_ylabel('Energy ($cm^{-1}$)', fontsize=20)

    fig.canvas.draw()
    fig.canvas.flush_events()

    if os.path.exists(args.output):
        os.remove(args.output)

    fig.savefig(args.output)
