import argparse
import glob
import os
from shutil import rmtree, copyfile
import subprocess


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
VASP_DIR = os.path.join(DATA_DIR, 'vasp')
OUT_DIR = os.path.join(DATA_DIR, 'primitive')
EXTRA_DIR = os.path.join(OUT_DIR, 'extra_data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restart', action='store_true', help='Recomputes completed calculations')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'vasp', '*.vasp')))

    if not os.path.exists(EXTRA_DIR):
        os.makedirs(EXTRA_DIR)

    for file in files:
        name = os.path.split(file)[-1]

        out_file = os.path.join(OUT_DIR, name)
        out_dir = os.path.join(EXTRA_DIR, name.replace('.vasp', ''))

        if os.path.exists(out_file):
            if args.restart:
                rmtree(out_file)
                rmtree(out_dir)
            else:
                continue
        
        os.makedirs(out_dir)
        copyfile(file, os.path.join(out_dir, 'POSCAR'))
        os.chdir(out_dir)

        subprocess.run(['phonopy', '--symmetry'])

        os.rename(os.path.join(out_dir, 'PPOSCAR'), out_file)
        os.chdir(DATA_DIR)

    print('FINISHED')
