from __future__ import annotations

import argparse
import csv
import re
import os
from typing import TYPE_CHECKING

import numpy as np

from ccdc.search import TextNumericSearch
import nistchempy as nist
import pubchempy as pcp

if TYPE_CHECKING:
    from ccdc.crystal import Crystal


HOME_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(HOME_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
DIFFICULT_DIR = os.path.join(HOME_DIR, 'difficult')


def find_actual_results(results: list, name: str) -> list:
    out = []
    for r in results:
        if (name == r.entry.chemical_name.lower() or name in r.entry.synonyms) and r.entry.has_3d_structure:
            out.append(r)

    return out


def get_best_experimental_conditions(results: list):
    neutron, xray, unknown = [], [], []
    for result in results:
        if result.entry.radiation_source.lower() == 'neutron':
            neutron.append(result)
        elif result.entry.radiation_source.lower().replace('-', '') == 'xray':
            xray.append(result)
        else:
            unknown.append(result)

    mins, indices = [], []
    for values in [neutron, xray, unknown]:
        if values:
            idx = np.argmin([clean_t(result.entry.temperature) for result in values])
            indices.append(idx)
            mins.append(clean_t(values[idx].entry.temperature))
        else:
            indices.append(-1)
            mins.append(1e6)

    neutron_min, xray_min, unknown_min = mins

    if neutron_min <= xray_min:
        if neutron_min <= unknown_min + 100:
            return neutron[indices[0]]
        else:
            return unknown[indices[2]]

    elif neutron_min <= xray_min + 50:
        if neutron_min <= unknown_min + 100:
            return neutron[indices[0]]
        elif xray_min <= unknown_min + 50:
            return xray[indices[1]]
        else:
            return unknown[indices[2]]

    else:
        return xray[indices[1]] if xray_min < unknown_min else unknown[indices[2]]


def has_all_hydrogens(crystal: Crystal) -> bool:
    copy = crystal.copy()
    try:
        copy.add_hydrogens('missing')
    except RuntimeError:
        return False

    return crystal.to_string('cif') == copy.to_string('cif')


def clean_t(temperature: str | None) -> float:
    if temperature is None:
        return 300

    temperature = temperature.replace('at', '').replace('K', '').strip()
    if 'deg.C' in temperature:
        temperature = temperature.replace('deg.C', '').strip()
        temperature = 273.15 + float(temperature)

    try:
        return float(temperature)
    except ValueError:
        print(f'!!!!!!!!!!!!problematic temperature: {temperature} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return 300


def split_on_disorder(results: list) -> tuple[list, list]:
    no_disorder, disorder = [], []
    for r in results:
        if r.entry.has_disorder:
            disorder.append(r)
        else:
            no_disorder.append(r)

    return disorder, no_disorder


def get_best_result(actual_results):
    disorder, no_disorder = split_on_disorder(actual_results)

    if no_disorder:
        return get_best_experimental_conditions(no_disorder), False
    else:
        return get_best_experimental_conditions(disorder), True


def write_best_cif(actual_results: list, line: list[str], out_name: str) -> list[str]:
    with_hydrogens = [r for r in actual_results if has_all_hydrogens(r.crystal)]
    if with_hydrogens:
        best, disorder = get_best_result(with_hydrogens)
        crystal = best.crystal
    else:
        best, disorder = get_best_result(actual_results)
        crystal = best.crystal.copy()
        try:
            crystal.add_hydrogens('missing', True)
        except RuntimeError:
            crystal = best.crystal

    dir = DIFFICULT_DIR if disorder else DATA_DIR
    out_name = f'{out_name}_{best.entry.ccdc_number}'

    with open(os.path.join(dir, out_name + '.cif'), 'w') as out:
        out.write(crystal.to_string('cif'))

    line[3] = out_name
    line[4] = best.entry.radiation_source if best.entry.radiation_source else '?'
    line[5] = clean_t(best.entry.temperature) if best.entry.temperature is not None else '?'

    return line


def search_for_results(name: str, line: list[str]) -> list[str] | None:
    out_name = name.replace(' ', '_').replace('-', '_')

    query = TextNumericSearch()
    query.add_compound_name(name)
    result = query.search()

    if actual_results := find_actual_results(result, name):
        return write_best_cif(actual_results, line, out_name)
    else:
        return None


def get_deuterated(new_csv: list[list[str]], current_line: list[str]):
    for line in reversed(new_csv):
        if not line[2]:
            current_line[3] = line[3]
            current_line[4] = line[4]
            current_line[5] = line[5]
            break

    return current_line


def check_name(name: str) -> str | None:
    name = name.lower()
    for check in ['amorphous', 'crystalline', 'single crystal', 'powder', 'oriented', 'oxidised',
                  'reduced']:
        if check in name:
            return f'{check} is a difficult instruction for script'

    return None


def search_synonyms(name: str, line: list[str]) -> list[str] | None:
    s = nist.run_search(identifier=name, search_type='name')
    pubchem = pcp.get_compounds(name, 'name')

    names = set()
    if s.compounds:
        names.union(s.compounds[0].synonyms)

    if pubchem:
        names.union(pubchem[0].synonyms)
        names = [pubchem[0].iupac_name] + list(names)

    for synonym in names:
        try:
            synonym = synonym.lower()
        except AttributeError:
            continue

        if result := search_for_results(synonym, line):
            return result
    return None


def scrape(redo_failed: bool = True):
    new_csv = []
    count_added = 0
    with open(CSV_PATH, 'r') as f, open(os.path.join(DATA_DIR, 'new_data.csv'), 'w', newline='') as new_file:
        reader = csv.reader(f, delimiter=',')

        writer = csv.writer(new_file, delimiter=',')
        new_csv.append(next(reader))

        for line in reader:
            writer.writerow(new_csv[-1])
            print(line[0])

            if line[3] or (line[6] and not redo_failed and line[6][:6] != 'script'):  # CIF file or reason for no file
                new_csv.append(line)
                print('exists')
                continue

            if line[2]:
                new_csv.append(get_deuterated(new_csv, line))
                print('deuterated')
                continue

            if problem := check_name(line[0]):
                line[6] = problem
                new_csv.append(line)
                print('problematic name')
                continue

            name = line[0].replace('(TFXA)', '').strip()

            try:
                year = int(name[-4:])
            except ValueError:
                year = None

            if year is not None:
                new_csv.append(get_deuterated(new_csv, line))
                print('has year')
                continue

            if result := search_for_results(name.lower(), line):
                new_csv.append(result)
                count_added += 1
                print('FOUND')
                continue

            oxidations = re.findall(r'\([IV]+\)', name)
            if oxidations:
                for oxidation_number in oxidations:
                    name.replace(oxidation_number + ' ', '')

                if result := search_for_results(name.lower(), line):
                    new_csv.append(result)
                    count_added += 1
                    print('FOUND without oxidation number')
                    continue

            unknown_deuteration = re.findall(r'-?\(?N?-?D+[0-9]+\)?', name)
            if unknown_deuteration:
                for deuteration in unknown_deuteration:
                    name.replace(deuteration + ' ', '')

                if result := search_for_results(name.lower(), line):
                    new_csv.append(result)
                    count_added += 1
                    print('FOUND without deuteration')
                    continue

            if line[10]:
                if result := search_for_results(line[10].lower(), line):
                    new_csv.append(result)
                    count_added += 1
                    print('FOUND using other name')
                    continue

            if result := search_synonyms(name, line):
                new_csv.append(result)
                count_added += 1
                print('FOUND as synonym')
                continue

            line[6] = 'script failed to find'
            new_csv.append(line)
            print('FAILED')

        writer.writerow(new_csv[-1])

    print(count_added)
    return new_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for scraping data from the CSD database.')
    parser.add_argument('-rf', '--redo-failed', action='store_true',
                        help='Causes the previously failed compounds to be re-attempted.')
    args = parser.parse_args()

    changed = scrape(args.redo_failed)

    # with open(os.path.join(HOME_DIR, 'new_data.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(changed)
