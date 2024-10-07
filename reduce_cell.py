import argparse
import glob
from io import TextIOWrapper
import os
from shutil import rmtree, copyfile
import subprocess

from ase.io import read


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
VASP_DIR = os.path.join(DATA_DIR, 'vasp')
OUT_DIR = os.path.join(DATA_DIR, 'primitive')
EXTRA_DIR = os.path.join(OUT_DIR, 'extra_data')


class CustomFile(TextIOWrapper):
    def readline(self, size=-1):
        line = super().readline(size)
        split = line.split()
        lens = [len(val) < 3 for val in split]
        
        if all(lens) and 'D' in split:
            line = line.replace('D', 'H')
        return line


def read_vasp_d(filename: str):
    with open(filename, 'rb') as f:
        f = CustomFile(f)
        atoms = read(f, format='vasp')
    return atoms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    args = parser.parse_args()

    if args.restart:
        rmtree(OUT_DIR)
        os.makedirs(OUT_DIR)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'vasp', '*.vasp')))

    cif2cell_files, vesta_files = [], []
    for file in files:
        if file[-10:] == 'vesta.vasp':
            vesta_files.append(file)
        else:
            cif2cell_files.append(file)

    if not os.path.exists(EXTRA_DIR):
        os.makedirs(EXTRA_DIR)

    skipped_vesta = 0

    for cif2cell_file, vesta_file in zip(cif2cell_files, vesta_files):
        print('\n')
        try:
            cif2cell = read(cif2cell_file, format='vasp')
            vesta = read(vesta_file, format='vasp')
        except KeyError:
            cif2cell = read_vasp_d(cif2cell_file)
            vesta = read_vasp_d(vesta_file)

        if cif2cell == vesta:
            print('Files identical; skipping VESTA')
            files = [cif2cell_file]
        else:
            files = [cif2cell_file, vesta_file]
        space_group_number = []
        for i, file in enumerate(files):
            name = os.path.split(file)[-1]
            print(name)

            out_file = os.path.join(OUT_DIR, name)
            out_dir = os.path.join(EXTRA_DIR, name.replace('.vasp', ''))

            if os.path.exists(out_file):
                print('Skipping . . .')
                break

            os.makedirs(out_dir)
            copyfile(file, os.path.join(out_dir, 'POSCAR'))
            os.chdir(out_dir)

            result = subprocess.run(['phonopy', '--symmetry'], stdout=subprocess.PIPE)
            for line in result.stdout.decode().split('\n'):
                if 'space_group_number' in line:
                    space_group_number.append(int(line.strip().split()[-1]))
                    break
            else:
                raise Exception()

            # Abandon VESTA file if the space group is conserved
            if i == 1:
                if space_group_number[0] == space_group_number[1]:
                    skipped_vesta += 1
                    print(f'Space groups identical ({space_group_number[0]}); skipping VESTA.')

                    os.chdir(DATA_DIR)
                    rmtree(out_dir)
                    continue
                else:
                    print(f'Space groups different (cif2cell={space_group_number[0]}, vesta={space_group_number[1]}); '
                          f'keeping both files.')

            os.rename(os.path.join(out_dir, 'PPOSCAR'), out_file)
            os.chdir(DATA_DIR)

    print(f'FINISHED, skipped {skipped_vesta} vesta files')
