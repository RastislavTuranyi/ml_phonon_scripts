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

    bands = phonons.get_dispersion().split(btol=5.)

    fig = plot_1d(bands, ylabel='Energy (meV)', c='#E94D36', alpha=0.8)

    fig.savefig(args.output)
