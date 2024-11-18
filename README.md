# ML phonon calculation scripts

This repository is a collection of scripts for running phonon calculations using machine-learned interatomic potentials
(MLIP). It includes scripts for the entire pipeline, from converting CIF files to something more useable, to the 
phonon calculations themselves.

All the scripts are based on top of the [janus-core](https://stfc.github.io/janus-core/index.html) package.

The pipeline consists of the scripts in the following order:

1. `convert_to_xyz.py` for converting CIF files obtained from crystal structure databases such as CDS/ICSD/etc. into the 
much more computer-friendly (and phonopy compatible) VASP POSCAR format.
2. `reduce_cell.py` for using phonopy to reduce each system into its primitive cell
3. `optimise.py` for optimising the structure using a given MLIP
4. `run_phonon.py` for running the phonon calculations on the optimised structure
5. `analyse_phonons.py` for checking whether phonon calculations concluded successfully
6. `plot_abins.py` for computing INS spectra from the phonon calculations and plotting the comparison to experimental spectra