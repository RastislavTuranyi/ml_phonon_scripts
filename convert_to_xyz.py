import glob
import os
import subprocess

from ase.io import read, write
from ase.io.cif import read_cif


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(path, '*.cif'))

    out_path = os.path.join(path, 'vasp')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for file in files:
        name = os.path.split(file)[-1]
        print(name)

        subprocess.run(['python', '-m', 'cif2cell',
                        '-f', file,
                        '-p', 'vasp',
                        '-o', os.path.join(out_path, 'temp.vasp'),
                        '--cartesian', '--vasp-cartesian',
                        '--no-reduce'])

        try:
            with open(os.path.join(out_path, 'temp.vasp'), 'r') as f:
                comment = f.readline()
                elements = comment.split('Species order:')[-1].strip()
                table1 = [f.readline() for _ in range(4)]
                rest = [line for line in f]
        except FileNotFoundError:
            continue

        with open(os.path.join(out_path, name.replace('.cif', '.vasp')), 'w') as f:
            f.write(comment)
            f.writelines(table1)
            f.write(elements + '\n')
            f.writelines(rest)

        os.remove(os.path.join(out_path, 'temp.vasp'))

        # write(os.path.join(out_path, name.replace('.cif', '.xyz')),
        #       data)