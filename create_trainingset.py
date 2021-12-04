#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create a moltemplate for
tubes with varying cross sections

@author: Javier E. Santos
Applied Machine Learning Summer School 2019
Active Learning for Nanopores Project
"""




import numpy as np
from matplotlib import pyplot as plt
import pore_utils as pore #imports my library
import pandas as pd
#np.random.seed(seed=1)


"""
User inputs
"""

sys_name = 'jesantos' #replace to Darwin if running in the cluster
#sys_name = 'Darwin' #replace to Darwin if running in the cluster

# Create random examples
num_cases = 100000000 #number of examples to run


# General

cs_shapes = ['ellipse', 'rectangle', 'pie'] #supported cross-sections



# Methane properties
#gas_density = 200 #[kg/m3] of the bulk. The code selects a random density
molecular_spacing = 3.73 #for methane
molar_mass = 16.04 #[g/mol]

# Simulation specs
NVT_ts = 500010

# Domain size
y_molecules = 32 #Cross section axis 1
z_molecules = 32 #Cross section axis 2
tube_length = 100 # num of molecules on the periodic dir

"""
This section generates random numbers to create varying shapes and sizes
for the simulation domain
"""

dy = np.arange(4,y_molecules-2) #possible range of cross-section
dz = np.arange(4,z_molecules-2)

for case in range(0,num_cases): #loop to run cases

    cross_section_shape = cs_shapes[ np.random.randint(0,np.shape(cs_shapes)[0]) ] #grabs a random cross-section

    gas_density = np.random.randint(50,200) #random [kg/m3] density of the bulk

    ys = np.arange(1,y_molecules)
    zs = np.arange(1,z_molecules)

    dy1 = dy[ np.random.randint(0,dy.shape[0]) ] #pick a random r1
    dz1 = dz[ np.random.randint(0,dz.shape[0]) ] #pick a random r2

    y_allowed = (ys+dy1)
    z_allowed = (zs+dz1)
    y_allowed = ys[ y_allowed< ys.max() ]
    z_allowed = zs[ z_allowed< zs.max() ]

    y1 = np.random.choice(y_allowed)  # first coordinate of the box that bounds the
    z1 = np.random.choice(z_allowed)  # cross-section

    y2 = y1 + dy1 #second coordinate
    z2 = z1 + dz1

    ang1 = np.random.randint(0,300)     #angles for pie slice
    ang2 = np.random.randint(ang1,360)  #angles for pie slice


    file_name = cross_section_shape + '_' + \
                str(y1) + '_' + str(y2) + '_' + \
                str(z1) + '_' + str(z2) + '_rho_' + \
                str(gas_density) #the folder is named as y1_y2_z1_z2_rhoBulk


    """
    Internals. Change the paths accordingly.
    """

    s_dirs_jesantos = {
            'name': 'jesantos',
            'molecules':     '../../', #location where methane.lt and trappe1998.lt are stored
            'moltemplate': 'moltemplate.sh',
            'packmol': '~/Programs/packmol/packmol-18.169/packmol',
            'lammps': '~/Programs/lammps/src/lmp_serial',
            'mpi': 'openmpi/3.1.2-intel_17.0.6'
            }

    s_dirs_Darwin = {
            'name': 'Darwin',
            'molecules':     '../../', #location where methane.lt and trappe1998.lt are stored
            'moltemplate': 'moltemplate.sh',
            'packmol': '/projects/lammps_for_al/lammps/tools/packmol/packmol',
            'lammps': '/projects/lammps_for_al/lammps_installed/bin/lmp',
            'mpi': 'openmpi/3.1.2-intel_17.0.6'
            }

    s_dirs_jesantosUT = {
            'name': 'jesantos',
            'molecules':     '../../', #location where methane.lt and trappe1998.lt are stored
            'moltemplate': 'moltemplate.sh',
            'packmol': '~/Programs/packmol/packmol',
            'lammps': 'lmp_daily',
            'mpi': 'openmpi/3.1.2-intel_17.0.6'
            }

    if sys_name == 'jesantos': s_directories = s_dirs_jesantosUT
    #if sys_name == 'jesantos': s_directories = s_dirs_jesantos
    if sys_name == 'Darwin'  : s_directories = s_dirs_Darwin

    ###################End of user inputs: proceed with caution (!)




    s_directories['example_family'] = cross_section_shape
    s_directories['example_name']   = file_name



    s_directories['quick_route'] = s_directories['example_family'] + '/' + \
                                   s_directories['example_name']

    pore.create_folders(s_directories) #this function creates the dirs and copies files


    x_size = molecular_spacing*tube_length
    y_size = molecular_spacing*y_molecules
    z_size = molecular_spacing*z_molecules


    print('The domain size is: %.1f x %.1f x %.1f'
          % (x_size/10, y_size/10, z_size/10), '[nm]')

    ################# Drawing and extruding cross-section

    if cross_section_shape == 'pie':
        np.savetxt(s_directories['quick_route']+'/Setup/angles.txt',
                   (ang1,ang2), header='angles')

    # Notes:
    # the x-coordiante is the periodic axis
    # The constant cross section will be defined by YZ



    cross_section, vol_3d = pore.create_cross_section_and3D( y_molecules,
                                                             z_molecules,
                                                             tube_length,
                                                             cross_section_shape,
                                                             (y1,y2),(z1,z2),
                                                             ang1=ang1,
                                                             ang2=ang2)


    ############## Molecular operations

    coords = pore.get_solid_coords(vol_3d, molecular_spacing)
    pore.write_moltemplate_file(coords, vol_3d, molecular_spacing, s_directories)
    pore.run_moltemplate(s_directories, 'solid')

    num_molecules = pore.compute_num_molecules(gas_density, molar_mass,
                                               molecular_spacing, vol_3d)

    pore.write_packmol_file(vol_3d, molecular_spacing, num_molecules,
                            s_directories)

    pore.run_packmol(s_directories)


    num_solid_mol = pore.write_final_coordinates(vol_3d, molecular_spacing,
                                                 s_directories,
                                                 cross_section_shape)

    pore.run_moltemplate(s_directories, 'all')

    pore.create_Minimization_files(s_directories, num_solid_mol, bcs='p p p')

    pore.run_lammps('Minimization', s_directories)

    pore.create_NVT_files(s_directories, 1, num_solid_mol, 'first_ts')
    pore.run_lammps('NVT', s_directories)
    pore.create_NVT_files(s_directories, NVT_ts, num_solid_mol, 'normal')
    pore.run_lammps('NVT', s_directories)
