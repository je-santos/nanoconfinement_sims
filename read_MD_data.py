#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:30:29 2019

@author: jesantos
"""

import os
import numpy as np
import pore_utils as pore
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt as e_dist


rho_epsilon = 0.01 #cut-off to calculate mean density

shapes = ['pie','rectangle','ellipse']

example_variables = ['cross_sec','input_mask','rho_after_min_32', 'porosity',
                         'rho_mean_after_min_32', 'e_dist',
                         'rho_after_min_128',
                         'y1','y2','z1','z2','rho_after_nvt_32','rho_after_nvt_128',
                         'rho_prescribed','rho_difference_percent'] #
    
all_example_data ={v:[] for v in example_variables}

for cross_sec in shapes:

    all_dirs = os.listdir( cross_sec+'/')
    
    for file_name in all_dirs:
        print(file_name)
        
        
        if os.path.isfile(file_name):
            print(f"not a directory: {file_name}")
            continue
        elif os.path.exists(cross_sec + '/' + file_name +
                          '/NVT/density_gas_final.profile') == False:
            print('Simulation currupted or on-going')
            continue
         
        
        all_example_data['cross_sec'].append(cross_sec)
        
        vol_3d_bin = np.load( '%s/%s/Setup/3D_domain.npy' % (cross_sec, file_name) )
        
        binary_cross_section = vol_3d_bin[0,:,:]
        
        all_example_data["input_mask"].append( np.int8(binary_cross_section) )
        
        e_dist_mask = np.float16( e_dist(binary_cross_section) )
        all_example_data["e_dist"].append( e_dist_mask )
        
        phi = np.sum(binary_cross_section)/np.size(binary_cross_section)*100
        
        all_example_data['porosity'].append(phi)
        
        init_rho_128 = pore.read_lammps_output('%s/%s'% (cross_sec, file_name),
                                           'initial', 1)
        all_example_data['rho_after_min_128'].append(init_rho_128*1000)
        
        init_rho_32 = pore.read_lammps_output('%s/%s'% (cross_sec, file_name),
                                           'initial', 4)
        
        init_rho_32 = init_rho_32*binary_cross_section #remove outside noise
        
        
        all_example_data['rho_after_min_32'].append(init_rho_32*1000)
        
        dens_in  = init_rho_32[ init_rho_32>rho_epsilon ].mean()
        
        all_example_data['rho_mean_after_min_32'].append(dens_in*binary_cross_section*1000)
   
        
        final_rho_128 = pore.read_lammps_output('%s/%s'% (cross_sec, file_name),
                                                'final',1)
        
        
        all_example_data['rho_after_nvt_128'].append(final_rho_128*1000)
        
        final_rho_32 = pore.read_lammps_output('%s/%s'% (cross_sec, file_name),
                                                'final',4)
        
        final_rho_32 = final_rho_32*binary_cross_section #remove outside noise
        
        all_example_data['rho_after_nvt_32'].append(final_rho_32*1000)
        
        
        all_example_data['y1'].append(file_name.split('_')[1])
        all_example_data['y2'].append(file_name.split('_')[2])
        all_example_data['z1'].append(file_name.split('_')[3])
        all_example_data['z2'].append(file_name.split('_')[4])
        
        all_example_data['rho_prescribed'].append(file_name.split('_')[6])
        
        
        
        
        dens_out = final_rho_32[final_rho_32>rho_epsilon].mean()
        disc_rho = np.abs(dens_in-dens_out)/dens_in*100
        
        all_example_data['rho_difference_percent'].append(disc_rho)
        
        
        print('The density error discrepancy is %f %%' % disc_rho)
        
    


df_MD = pd.DataFrame.from_dict(all_example_data)
df_MD.to_pickle('Results/shapes_training_data_newUT.pkl')
        