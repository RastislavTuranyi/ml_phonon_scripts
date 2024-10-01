"""
Script for converting CIF files to VASP's POSCAR files. This is a neccessary step because the CIF file format was
designed more with human-readability in mind rather than machine-readability, and is therefore tricky to parse and use.
In fact, two different systems are used for this conversion - cif2cell and VESTA - to make sure that each CIF file was
converted successfully.

The output format used is POSCAR because, despite its naming convention, it if well defined and allows for good
definition of a periodic material. Furthermore, phonopy, which is the package that is used later on for the phonon
calculations, was originally built around VASP, making the POSCAR format the most suitable for use with phonopy.
"""
import argparse
import glob
import os
import platform
import subprocess

# from ase.io import read


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(path, 'vasp')

    parser = argparse.ArgumentParser(description='Script for convertin CIF files to VASP POSCAR files.')
    parser.add_argument('-i', '--input', type=str, default=path,
                        help='The path to the directory containing the CIF files to convert. Defaults to the directory'
                             ' containing this script.')
    parser.add_argument('-o', '--output', type=str, default=out_path,
                        help='The path to which to save the converted files. Defaults to the subdirectory "vasp" inside'
                             ' the input directory')
    parser.add_argument('-v', '--vesta', type=str,
                        default='vesta.exe' if platform.system().lower() == 'windows' else 'vesta',
                        help='The path to the VESTA executable. Defaults to just "vesta" (or "vesta.exe" on Windows), '
                             'assuming that the program is on PATH.')
    parser.add_argument('-dc', '--disable-cif2cell', action='store_true',
                        help='If provided, POSCAR files are not generated using cif2cell.')
    parser.add_argument('-dv', '--disable-vesta', action='store_true',
                        help='If provided, POSCAR files are not generated using VESTA.')
    args = parser.parse_args()

    path = args.input
    out_path = args.output
    files = glob.glob(os.path.join(path, '*.cif'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for file in files:
        name = os.path.split(file)[-1]
        print(name)

        if not args.disable_cif2cell:
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

            out_cif2cell = os.path.join(out_path, name.replace('.cif', '.vasp'))
            with open(out_cif2cell, 'w') as f:
                f.write(comment)
                f.writelines(table1)
                f.write(elements + '\n')
                f.writelines(rest)

            os.remove(os.path.join(out_path, 'temp.vasp'))

        if not args.diable_vesta:
            name2 = name.replace('.cif', '_vesta.vasp')
            out_vesta = os.path.join(out_path, name2)
            subprocess.run([args.vesta, '-nogui',
                            '-i', file,
                            '-merge_split_site',
                            '-save', 'format=VASP', 'option=cartesian', out_vesta,
                            ])
