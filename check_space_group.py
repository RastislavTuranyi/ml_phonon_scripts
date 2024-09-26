import argparse
import glob
import os


TOP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.join(TOP_DIR, 'data', 'optimised')
CELL_DIR = os.path.join(HOME_DIR, 'cell')
NO_DIR = os.path.join(HOME_DIR, 'no_cell')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cell', action='store_true', help='If provided, the cell parameters are optimised')
    args = parser.parse_args()

    target_dir = CELL_DIR if args.cell else NO_DIR

    files = sorted(glob.glob(os.path.join(target_dir, 'extra_data', '*', '*-log.yml')))
    #print(os.path.join(target_dir, 'extra_data', '*', '*-log.yaml'))
    results = []
    for file in files:
        before, after = None, None
        with open(file, 'r') as f:
            for line in f:
                #print(line)
                if 'Before optimisation spacegroup:' in line:
                    before = line.split('Before optimisation spacegroup:')[-1].replace('"', '').strip()
                elif 'After optimization spacegroup:' in line:
                    after = line.split('After optimization spacegroup:')[-1].replace('"', '').strip()
                    break
            else: 
                raise Exception()

        ok = 'OK' if before == after else 'FAIL'
        name = os.path.split(os.path.dirname(file))[-1]
        print(f'{ok}    {name}   before: {before};  after: {after}')
        
        results.append(', '.join([name, str(before == after), before, after]))
        if before == after:
            name = 'spacegroup_conserved'
        else:
            name = 'spacegroup_changed'

        with open(os.path.join(os.path.dirname(file), name), 'w') as f:
            f.write(before + '   ' + after)

    with open(os.path.join(target_dir, 'spacegroup_check.csv'), 'w') as f:
        f.writelines(results)
