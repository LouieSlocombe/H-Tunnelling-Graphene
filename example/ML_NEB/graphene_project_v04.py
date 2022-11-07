import copy
import os
import time

import ase.calculators.castep
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.lattice.hexagonal import Graphene
from ase.visualize import view

from ase.io import read
import numpy as np
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.io import read
import matplotlib.pyplot as plt
from catlearn.optimize.mlneb import MLNEB
from ase.neb import NEBTools
from ase.neb import NEB
import copy
from catlearn.optimize.tools import plotneb


def castep_calc(atoms, xc_f="PBE", e_cut=800, kpts=[4, 4, 1], directory='data', ps_type='ext', ps_ext='.usp'):
    calc = ase.calculators.castep.Castep(keyword_tolerance=1)

    # include interface settings in .param file
    calc._export_settings = True
    calc._pedantic = True

    # reuse the same directory
    calc._seed = directory
    calc._label = directory
    calc._directory = directory
    calc._rename_existing_dir = False

    # calc._link_pspots = True
    # calc._copy_pspots = False
    # calc._build_missing_pspots = False

    # Param settings
    calc.param.reuse = True
    calc.param.xc_functional = xc_f
    calc.param.cut_off_energy = e_cut
    calc.param.finite_basis_corr = 0
    calc.param.num_dump_cycles = 0  # Prevent CASTEP from writing *wvfn* files
    calc.param.write_checkpoint = 'minimal'
    calc.param.max_scf_cycles = 100
    calc.param.opt_strategy = "speed"
    calc.param.mixing_scheme = "Broyden"  # Broyden Pulay
    calc.param.spin_polarised = False

    calc.param.sedc_apply = True
    calc.param.sedc_scheme = 'MBD*'
    calc.param.perc_extra_bands = 50
    calc.param.elec_energy_tol = 1e-6

    # Cell settings
    calc.cell.kpoint_mp_grid = kpts
    calc.cell.fix_com = False
    calc.cell.symmetry_tol = 0.001
    calc.cell.fix_all_cell = True
    # calc.cell.symmetry_generate = True

    ele_list = atoms.get_chemical_symbols()
    tmp = None
    ps_suff = None
    if xc_f == 'PBE' or xc_f == 'PBE0':
        tmp = 'PBE'
    elif xc_f == 'B3LYP' or xc_f == 'BLYP':
        tmp = 'BLYP'
    elif xc_f == 'LDA':
        tmp = 'LDA'
    else:
        exit('problem with xc and not supporting ps type...')
    ps_type = ps_type.upper()
    if ps_type == 'NCP':
        ps_suff = '_%s19_%s_OTF%s' % (ps_type, tmp, ps_ext)
    elif ps_type == 'SP':
        ps_suff = '_00%s' % ps_ext
    elif ps_type == 'EXT':
        ps_suff = '_%s_%s_OTF%s' % (ps_type, tmp, ps_ext)
    else:
        exit('problem with ps type...')

    for e in ele_list:
        calc.cell.species_pot = (e, e + ps_suff)

    return calc


def castep_calc_temp(template, xc_f="PBE", e_cut=800, kpts=[4, 4, 1], directory='data', ps_type='ext', ps_ext='.usp'):
    atoms = read(template)
    calc = atoms.calc

    # include interface settings in .param file
    calc._export_settings = True
    calc._pedantic = True

    # reuse the same directory
    calc._seed = directory
    calc._label = directory
    calc._directory = directory
    calc._rename_existing_dir = False

    # calc._link_pspots = True
    # calc._copy_pspots = False
    # calc._build_missing_pspots = False

    # Param settings
    calc.param.reuse = True
    calc.param.xc_functional = xc_f
    calc.param.cut_off_energy = e_cut
    calc.param.finite_basis_corr = 0
    calc.param.num_dump_cycles = 0  # Prevent CASTEP from writing *wvfn* files
    calc.param.write_checkpoint = 'minimal'
    calc.param.max_scf_cycles = 100
    calc.param.opt_strategy = "speed"
    calc.param.mixing_scheme = "Broyden"  # Broyden Pulay
    calc.param.spin_polarised = False
    # calc.param.elec_method = "EDFT"

    calc.param.sedc_apply = True
    calc.param.sedc_scheme = 'MBD*'
    calc.param.perc_extra_bands = 50
    calc.param.elec_energy_tol = 1e-6

    ele_list = atoms.get_chemical_symbols()
    tmp = None
    ps_suff = None
    if xc_f == 'PBE' or xc_f == 'PBE0':
        tmp = 'PBE'
    elif xc_f == 'B3LYP' or xc_f == 'BLYP':
        tmp = 'BLYP'
    elif xc_f == 'LDA':
        tmp = 'LDA'
    else:
        exit('problem with xc and not supporting ps type...')
    ps_type = ps_type.upper()
    if ps_type == 'NCP':
        ps_suff = '_%s19_%s_OTF%s' % (ps_type, tmp, ps_ext)
    elif ps_type == 'SP':
        ps_suff = '_00%s' % ps_ext
    elif ps_type == 'EXT':
        ps_suff = '_%s_%s_OTF%s' % (ps_type, tmp, ps_ext)
    else:
        exit('problem with ps type...')

    for e in ele_list:
        calc.cell.species_pot = (e, e + ps_suff)

    return calc


def make_graphene_sheet(i1=4, i2=4, alat=2.45, vaccum=12.8):
    # Set up a graphene lattice with a i1xi2 supercell
    return Graphene(symbol='C', latticeconstant={'a': alat, 'c': vaccum}, size=(i1, i2, 1))


def make_h_adsorbate(input_atoms, index=11, height=1.5, pos=None):
    if pos is None:
        # Get the site to adsorb the H atom
        pos = input_atoms.get_positions()
        pos = pos[index]
    # Make a hydrogen atom
    h_atom = Atoms('H', positions=[(pos[0], pos[1], pos[2] + height)])
    # Add a hydrogen atom to the slab
    return input_atoms + h_atom


def file_list(mypath=os.getcwd()):
    """
    List only the files in a directory given by mypath
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of files
    """
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles


# List only files which contain a substring
def sub_file_list(mypath, sub_str):
    """
    List only files which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of files which have been filtered
    """
    return [i for i in file_list(mypath) if sub_str in i]


f_max = 0.01
uncert_max = 0.05
n_ml = 1000

f_debug = None
f_spe = False
f_resub = False
f_optimise = False
f_mlneb = True
f_neb = False
f_ts_refine = False

ml_neb_images = 20
neb_images = 10
interp = 'idpp'  # Choose between linear or idpp interpolation or a list of images
f_sequential = False
f_print_full = True

ml_neb_traj_file = r'ML-NEB.traj'

print('Starting program...', flush=True)

# Check if the ML-NEB trajectory file exists
if "last_predicted_path.traj" in sub_file_list(os.getcwd(), '.traj'):
    print("Found last_predicted_path.traj", flush=True)
    # Set to resubmit the ML-NEB calculation
    f_resub = True

os_name = os.name
if os_name == 'nt':
    # Windows
    f_debug = True
else:
    from sella import Sella
    from sella import IRC

    # Linux
    f_debug = False

# reactant = ase.io.castep.read_castep("4x4.castep")
# calc = castep_calc_temp("4x4-out.cell")

if f_spe:
    print("Running single point energy calculation", flush=True)
    reactant = read("reactant.cell")  # react.cell 4x4-out.cell
    # Apply the calculator
    calc = castep_calc(reactant)
    # Single point calculation
    reactant.set_calculator(copy.deepcopy(calc))
    t0 = time.time()
    enegy = reactant.get_potential_energy()
    t1 = time.time()
    print('Time for energy calculation: %f s' % (t1 - t0))
    print('Energy: %f eV' % enegy)
    print("Finished single point energy calculation", flush=True)

if f_optimise is True and f_mlneb is False and f_ts_refine is False:
    reactant = read("reactant.cell")  # react.cell 4x4-out.cell
    # Apply the calculator
    calc = castep_calc(reactant)
    print('Optimising the inputs...', flush=True)
    # Set the calculator
    reactant.set_calculator(copy.deepcopy(calc))
    # Optimise the structure
    dyn = BFGS(reactant, trajectory='reactant.traj', logfile='reactant.log')
    dyn.run(fmax=f_max)
    print('Optimisation complete', flush=True)

if f_mlneb is True and f_ts_refine is False:
    print('Running ML-NEB', flush=True)
    reactant = read("reactant.cell")  # react.cell 4x4-out.cell
    product = read("product.cell")  # read("product.cell")
    # Apply the calculator
    calc = castep_calc(reactant)
    # Optimise the input structures
    if f_resub is False and f_optimise is True:
        print('Optimising the inputs...', flush=True)
        # Apply the calculator and optimise
        reactant.set_calculator(copy.deepcopy(calc))
        dyn = BFGS(reactant, trajectory='reactant.traj', logfile='reactant.log')
        dyn.run(fmax=f_max)

        product.set_calculator(copy.deepcopy(calc))
        dyn = BFGS(product, trajectory='product.traj', logfile='product.log')
        dyn.run(fmax=f_max)

        reactant = read('reactant.traj', index=-1)
        product = read('product.traj', index=-1)

    # Attach calculators
    reactant.set_calculator(copy.deepcopy(calc))
    product.set_calculator(copy.deepcopy(calc))

    # Get energy
    print('Reactant energy = ', reactant.get_potential_energy(), flush=True)
    print('Product energy = ', product.get_potential_energy(), flush=True)

    # Make sure forces are known
    print('Max forces on reactant = ', np.max(reactant.get_forces()), flush=True)
    print('Max forces on product = ', np.max(product.get_forces()), flush=True)

    print('Running ML NEB!', flush=True)
    # Prepare the ML NEB object
    neb_catlearn = MLNEB(start=reactant,  # reac reac_traj
                         end=product,  # prod prod_traj
                         ase_calc=calc,
                         n_images=ml_neb_images,  # Number of images (integer or float)
                         interpolation=interp,  # Choose between linear or idpp interpolation
                         restart=f_resub)  # Resubmission flag

    # Run the Ml NEB calculation
    neb_catlearn.run(fmax=f_max,  # Convergence criteria (in eV/Angs).
                     trajectory=ml_neb_traj_file,
                     full_output=f_print_full,  # Whether to print on screen the full output
                     unc_convergence=uncert_max,  # Maximum uncertainty for convergence (in eV).
                     ml_steps=n_ml,  # Maximum number of steps for the NEB on the predicted landscape.
                     sequential=f_sequential,  # One moving image to find the saddle first
                     )
    print('Done!', flush=True)
    plotneb(trajectory=ml_neb_traj_file, view_path=False)

if f_ts_refine is True and f_mlneb is False:
    ts_image = read("4x4-out.cell")
    calc = castep_calc(ts_image)
    print('TS refine', flush=True)
    # Attach the calculator to the TS image
    ts_image.set_calculator(copy.deepcopy(calc))
    # Get the forces
    energy = ts_image.get_potential_energy()
    print('Initial energy: ', energy, flush=True)
    forces = ts_image.get_forces()
    print('Initial max force: ', np.max(forces), flush=True)

    # Optimise the TS image

    print('Running Sella TS search...', flush=True)
    # Run Sella TS search
    sella_ts = Sella(ts_image,
                     logfile='sella_ts.log',
                     trajectory='sella_ts.traj')
    sella_ts.run(fmax=f_max, steps=1000)
    print('Done!', flush=True)

if f_neb is True and f_ts_refine is False:
    print('Running NEB', flush=True)
    reactant = read("reactant.cell")  # react.cell 4x4-out.cell
    product = read("product.cell")  # read("product.cell")
    # Apply the calculator
    calc = castep_calc(reactant)
    # Optimise the input structures
    if f_resub is False and f_optimise is True:
        print('Optimising the inputs...', flush=True)
        # Apply the calculator and optimise
        reactant.set_calculator(copy.deepcopy(calc))
        dyn = BFGS(reactant, trajectory='reactant.traj', logfile='reactant.log')
        dyn.run(fmax=f_max)

        product.set_calculator(copy.deepcopy(calc))
        dyn = BFGS(product, trajectory='product.traj', logfile='product.log')
        dyn.run(fmax=f_max)

        reactant = read('reactant.traj', index=-1)
        product = read('product.traj', index=-1)

    # Attach calculators
    reactant.set_calculator(copy.deepcopy(calc))
    product.set_calculator(copy.deepcopy(calc))

    # Get energy
    print('Reactant energy = ', reactant.get_potential_energy(), flush=True)
    print('Product energy = ', product.get_potential_energy(), flush=True)

    # Make sure forces are known
    print('Max forces on reactant = ', np.max(reactant.get_forces()), flush=True)
    print('Max forces on product = ', np.max(product.get_forces()), flush=True)

    print('Running NEB!', flush=True)
    # Prepare the band path
    images = [reactant]
    # loop over the number of images
    for i in range(neb_images - 2):
        images += [reactant.copy()]
    # Add the product
    images += [product]
    # Make NEB object
    neb = NEB(images)
    # Make interpolation
    neb.interpolate(method=interp)

    # Attach the calculator
    for image in images:
        image.set_calculator(copy.deepcopy(calc))
    # Set up the optimiser
    optimizer = BFGS(neb, trajectory='NEB.traj', logfile='NEB.log')
    # Run the optimisation
    optimizer.run(fmax=f_max)
