#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:26:06 2019

@author: jesantos
"""

import numpy as np
import scipy.constants
import subprocess
import os
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label as find_conn_comps 
#from sklearn.metrics import pairwise_distances as distances
from PIL import Image, ImageDraw #modules to draw diff shapes
import pandas as pd


def run_lammps(run_type, s_dirs):
    
    """
    Calls lammps to perform either Min or NVT
    
    """
    
    
    if run_type == 'Minimization':
        
        if s_dirs['name'] == 'Darwin':
            subprocess.run('module load intel/17.0.6', 
                       shell=True,capture_output=True,check=False)
            subprocess.run('module load mpich/3.2.1-intel_17.0.6', 
                       shell=True,capture_output=True,check=False)
            
            proc = subprocess.run(
           ('cd %s/Minimization && mpirun %s -i run.in.min') % 
           (s_dirs['quick_route'],s_dirs['lammps']), 
            shell=True,capture_output=True,check=False )
        else:
            proc = subprocess.run(
           ('cd %s/Minimization && %s -i run.in.min') % 
           (s_dirs['quick_route'],s_dirs['lammps']), 
            shell=True,capture_output=True,check=False )
    
        
    if run_type == 'NVT':
        
        #export OMP_NUM_THREADS=8
        
        if s_dirs['name'] == 'Darwin':
            proc = subprocess.run(
           ('cd %s/NVT && mpirun %s -i run.in.nvtnve') % 
           (s_dirs['quick_route'],s_dirs['lammps']), 
            shell=True,capture_output=True,check=False )
        else:
            proc = subprocess.run(
           ('cd %s/NVT && %s -i run.in.nvtnve') % 
           (s_dirs['quick_route'],s_dirs['lammps']), 
            shell=True,capture_output=True,check=False )
    
    
    print( proc )
    

def create_Minimization_files(s_dirs, num_solid_mol, bcs='p p f'):
    """
    Prints the default styles and the input file for the
    minimization run. The bcs can be specified by the user
    
    """
    
    
    with open(s_dirs['quick_route'] + '/Minimization/Default_styles.init','w+') as input_file:
        print('# -- Default styles for "TraPPE" -- \n \
              \tunits           real \n \
              \tatom_style      full \n \
              \t# (Hybrid force field styles were used for portability.) \n \
              \tbond_style      hybrid harmonic \n \
              \tangle_style     hybrid harmonic \n \
              \tdihedral_style  hybrid opls \n \
              \tpair_style      hybrid lj/charmm/coul/charmm 9.0 11.0 9.0 11.0 \n \
              \tpair_modify     mix arithmetic \n \
              \tspecial_bonds   lj 0.0 0.0 0.0', file=input_file)
        print('boundary',bcs, file=input_file)
    
    with open(s_dirs['quick_route'] + '/Minimization/run.in.min','w+') as input_file:
        print('include   "Default_styles.init"', file=input_file)
        print('read_data "../Setup/', s_dirs['example_name'], '_all_molecules.data"', sep='', file=input_file)
        print('include   "final.in.settings"', file=input_file)
        print( ('thermo        500  \n \
                 group walls molecule < %d  \n \
                 group full molecule > %d  \n \
                 fix freeze walls setforce 0.0 0.0 0.0  \n \
                 minimize 1.0e-4 1.0e-6 300000 800000  \n \
                 write_data   system_after_min.data') % 
            (num_solid_mol+1,num_solid_mol),file=input_file)

#compute        93 walls chunk/atom bin/2d y lower 0.9325 z lower 0.9325 units box \n \
#fix            94 walls ave/chunk 1 1 1 93 density/mass ave running file density_solid.profile \n \    
        
def create_NVT_files(s_dirs, ts, num_solid_mol, run_type):
    
    
    if run_type == 'normal':
    
        with open(s_dirs['quick_route'] + '/NVT/run.in.nvtnve','w+') as input_file:
            print('include   "../Minimization/Default_styles.init"', file=input_file)
            print('read_data ../Minimization/system_after_min.data', file=input_file)
            print('include   "../Minimization/final.in.settings"', file=input_file)
            
            print('timestep       1.0 \n \
                  group          wall molecule < %d \n \
                  group          full molecule > %d \n \
                  velocity full create 321 4928459 rot yes dist gaussian \n \
                  fix            freeze wall setforce 0.0 0.0 0.0 \n \
                  fix            fxnvt full nvt temp 321 321 10.0 tchain 1 \n \
                  fix            fxnve wall nve \n \
                  compute        check full stress/atom NULL \n \
                  compute        middletemp full temp \n \
                  compute        91 full chunk/atom bin/2d y lower 0.9325 z lower 0.9325 units box \n \
                  fix            92 full ave/chunk 10 10000 100000 91 density/mass ave running file density_gas_final.profile \n \
                  fix            93 full ave/chunk 10 10000 100000 91 c_check[1] c_check[2] c_check[3] ave running file pressure.profile \n \
                  compute        p all reduce sum c_check[1] c_check[2] c_check[3] \n \
                  variable       press equal -(c_p[1]+c_p[2]+c_p[3])/(3.*vol) \n \
                  thermo_style   custom step temp c_middletemp etotal press v_press  \n \
                  thermo         1000 \n \
                  #dump           1 all custom 50000 dump.nvtnve.methane_newstart.* id mol type x y z vx vy vz c_check[1] c_check[2] c_check[3] \n \
                  \n \
                  run		%d  \n \
                  write_data   system_after_nvt.data'
                  % (num_solid_mol+1, num_solid_mol, ts), file=input_file )
            
    elif run_type == 'first_ts':
        with open(s_dirs['quick_route'] + '/NVT/run.in.nvtnve','w+') as input_file:
            print('include   "../Minimization/Default_styles.init"', file=input_file)
            print('read_data ../Minimization/system_after_min.data', file=input_file)
            print('include   "../Minimization/final.in.settings"', file=input_file)
            
            print('timestep       0.0000001 \n \
                  group          wall molecule < %d \n \
                  group          full molecule > %d \n \
                  velocity full create 321 4928459 rot yes dist gaussian \n \
                  fix            freeze wall setforce 0.0 0.0 0.0 \n \
                  fix            fxnvt full nvt temp 321 321 10.0 tchain 1 \n \
                  fix            fxnve wall nve \n \
                  compute        check full stress/atom NULL \n \
                  compute        middletemp full temp \n \
                  compute        91 full chunk/atom bin/2d y lower 0.9325 z lower 0.9325 units box \n \
                  fix            92 full ave/chunk 1 1 1 91 density/mass ave running file density_gas_first_ts.profile \n \
                  compute        94 wall chunk/atom bin/2d y lower 0.9325 z lower 0.9325 units box \n \
                  fix            95 wall ave/chunk 1 1 1 94 density/mass ave running file density_wall_first_ts.profile \n \
                  compute        p all reduce sum c_check[1] c_check[2] c_check[3] \n \
                  variable       press equal -(c_p[1]+c_p[2]+c_p[3])/(3.*vol) \n \
                  thermo_style   custom step temp c_middletemp etotal press v_press  \n \
                  thermo         1000 \n \
                  #dump           1 all custom 50000 dump.nvtnve.methane_newstart.* id mol type x y z vx vy vz c_check[1] c_check[2] c_check[3] \n \
                  \n \
                  run		%d  \n \
                  write_data   system_after_nvt.data'
                  % (num_solid_mol+1, num_solid_mol, ts), file=input_file )
        
        
        
              
def create_cross_section_and3D(y,z, tube_l,figure, ys, zs, ang1=0, ang2=180, ap=10):
    """
    Draws a gemetric cross section which then is extruded to form a
    3D domain for simulation
    
    y and z are tuple with two coordinates that define the 2D figure
    tube_l is the number of molecules in depth
    
    figure is a string with one of the possible cross-sections
    
    ang1 and ang2 are optional arguments that define angles for certain features
    
    ap is the aperture of the parellel plate system
    
    """
    
    cross_section = Image.new('1', (y, z)) #create new image file
    draw          = ImageDraw.Draw( cross_section )
    
    
    if figure == 'ellipse':
        draw.ellipse( (ys[0], zs[0], ys[1], zs[1]), width=1, outline ='white')
    elif figure == 'rectangle':
        draw.rectangle( (ys[0], zs[0], ys[1], zs[1]), width=1, outline ='white')
    elif figure == '4side_polygon':
        draw.polygon( (ys[0], zs[0], ys[1], zs[1]), outline ='white')
    elif figure == 'pie':
        draw.pieslice( (ys[0], zs[0], ys[1], zs[1]), width=1, outline ='white', 
                                                          start=ang1, end=ang2)
    elif figure == 'plates':
        draw.line( (0,    0, y-1,    0), width=1, fill='white')
        draw.line( (0, ap+1, y-1, ap+1), width=1, fill='white')
        
    else:
        raise Exception('The cross section %s does not exist' % figure)
        
        

    cross_section = 1*np.array( cross_section ) #converts the image file to np array
    plt.figure;plt.imshow( cross_section ); plt.title('Cylinder cross-section')
    
    vol_3d = cross_section[np.newaxis,:,:]
    vol_3d = np.repeat(vol_3d,tube_l,axis=0) #extrudes the cross section in the x-dir
    
    

    return cross_section,vol_3d

    


def run_packmol(s_dirs):
    proc = subprocess.run(
           ('cd %s/Setup && %s < %s_free_gas.inp') % 
           (s_dirs['quick_route'],s_dirs['packmol'],s_dirs['example_name']), 
            shell=True,capture_output=True,check=False )
    print( proc )




def run_moltemplate(s_dirs, op, loc=None):
    
    if op == 'solid':
        
        if loc == 'local':
            proc = subprocess.run(
                    ('cd %s/Setup && moltemplate.sh %s_solid.lt') % 
                    (s_dirs['quick_route'],s_dirs['example_name']), 
                    shell=True,check=False )
            
            
        else:
            proc = subprocess.run(
                    ('cd %s/Setup && moltemplate.sh %s_solid.lt') % 
                    (s_dirs['quick_route'],s_dirs['example_name']), 
                    shell=True,capture_output=True,check=False )
            
    
    if op == 'all':
        proc = subprocess.run(
                ('cd %s/Setup && moltemplate.sh -xyz %s_all_molecules.xyz %s_all_molecules.lt') % 
                (s_dirs['quick_route'],s_dirs['example_name'],s_dirs['example_name'] ), 
                shell=True,capture_output=True,check=False )
        
        proc1 = subprocess.run(
                ('cp %s/Setup/*all_molecules* %s/Minimization/.') %
                (s_dirs['quick_route'], s_dirs['example_name']), 
                shell=True,capture_output=True,check=False )
    
    
    
    print(proc)
    

     

def compute_num_molecules(rho, molar_mass, mol_space, vol_3d):
    
    na  = scipy.constants.Avogadro #Avogadro num
    vol = vol_3d.size*mol_space**3
    
    num_molecules = rho*na*vol/molar_mass*(1000)*(1e-10)**3
    
    print('')
    print('%d molecules are needed to achieve the desired density' % num_molecules)
    
    return int(num_molecules)
    
    


def create_folders(s_dirs):
    
    file_name = s_dirs['quick_route']
    
    if not os.path.exists(file_name):
        os.makedirs(file_name + '/Setup')
        os.makedirs(file_name + '/Minimization')
        os.makedirs(file_name + '/NVT')
        print('Directory', file_name, 'created')
        shutil.copy('General_files/methane.lt',file_name + '/Setup/')
        shutil.copy('General_files/methane.xyz',file_name + '/Setup/')
        
        shutil.copy('General_files/final.in.settings',file_name +'/Minimization/')
        
    else:
        print('Directory' , file_name ,  'already exists')
        
           
    



def write_final_coordinates(vol_3d, mol_space, s_dirs, cs_shape):
    """
    Reads the packmol output file, and erases the molecules overlapping with
    the solid
    Writes the input coordinates for lammps Minimization
    
    The plates flag is used when a par of plates is used as domain
    
    Comments: Might not be general enough for any 3D geometry
    In simple cases, the central region is labeled as 2, this
    might have to be assessed case-by-case
    """
    
    if cs_shape == 'plates':
        plates = True
    else:
        plates = False
        
    # solid coordinates
    *_,xs,ys,zs = np.loadtxt(s_dirs['quick_route']+ '/Setup/' + s_dirs['example_name']+'_solid.data',
                             skiprows=25,unpack=True)
    # free gas coordinates
    xg,yg,zg    = np.loadtxt(s_dirs['quick_route']+ '/Setup/' + s_dirs['example_name'] + '_free_gas.xyz',
                             usecols=(1,2,3), skiprows=2,unpack=True)
                             
    # Obtain the voxelized coordinates of the molecules
    gas_indices = np.abs( np.trunc( (xg/mol_space, 
                                     yg/mol_space, 
                                     zg/mol_space) ).T.astype(int) )
    
    # If the molecules are a little outside our volume it brings them back
    gas_indices[:,0][ [gas_indices[:,0] > vol_3d.shape[0]-1] ] = vol_3d.shape[0]-1 
    gas_indices[:,1][ [gas_indices[:,1] > vol_3d.shape[1]-1] ] = vol_3d.shape[1]-1 
    gas_indices[:,2][ [gas_indices[:,2] > vol_3d.shape[2]-1] ] = vol_3d.shape[2]-1 
    

    # find connected components
    vol_3d_bis  = vol_3d + 1; vol_3d_bis[vol_3d_bis==2]=0 #changes 0's to 1's
    
    if plates == True: # closes the plates to identify the void space
        for i in range(1, vol_3d_bis.shape[1]):
            if vol_3d_bis[0,i,0] == 1:
                vol_3d_bis[:,i, 0] = 0
                vol_3d_bis[:,i,-1] = 0
            else: break
                
    
    filter_conn = np.zeros((3,3,3), dtype=np.int) #filter to find connected regions
    filter_conn[:,1,0]=1
    filter_conn[:,1,1]=1
    filter_conn[:,0,1]=1
    filter_conn[:,2,1]=1
    filter_conn[:,1,2]=1
    
    
    vol_3d_bis, ncomponents = find_conn_comps(vol_3d_bis, filter_conn)
    print(ncomponents,'void connected regions where found.\n')
    #plt.imshow(vol_3d_bis[0,:,:])
    
    geometry_center_label = 2 #WARNING: This might not hold for every case
                              #the main area of interested might be labed differently
                              #works for now. Maybe we could compare the areas
       

                       
    if plates == True:
        geometry_center_label = 1 # hard-coded, works so far
        
        for i in range(1, vol_3d_bis.shape[1]):
            if vol_3d_bis[0,i,0] == 0:
                vol_3d_bis[:,i, 0] = 1
                vol_3d_bis[:,i,-1] = 1
            else: break
        
        vol_3d_bis[:,i-1, 0] = 0
        vol_3d_bis[:,i-1,-1] = 0
        
        
        
    
    vol_3d_bis[ vol_3d_bis != geometry_center_label ] = 0
    vol_3d_bis = vol_3d_bis.astype('bool')
    
    plt.figure(); plt.imshow(vol_3d_bis[0,:,:]); plt.title('Pore system')
    plt.savefig(s_dirs['quick_route'] + '/Setup/Pore_system.png')
    plt.close()
    
    
    np.save('%s/Setup/3D_domain' % s_dirs['quick_route'],vol_3d_bis*1)
    
    valid_gas_mask = np.zeros(gas_indices.shape[0], dtype=bool)
    
    for i in range( 0, gas_indices.shape[0] ):
        valid_gas_mask[i] = vol_3d_bis[   gas_indices[i,0], 
                                          gas_indices[i,1], 
                                          gas_indices[i,2] ] 
    

    xg_valid, yg_valid, zg_valid =  xg[valid_gas_mask], \
                                    yg[valid_gas_mask], \
                                    zg[valid_gas_mask]
     
    
    #Write all the solid and valid gas molecule coordinates to a file
    x,y,z = np.concatenate( (xs,xg_valid) ), \
            np.concatenate( (ys,yg_valid) ), \
            np.concatenate( (zs,zg_valid) )
            
    num_gas_molecules = np.size(x)
    
    with open(s_dirs['quick_route'] + '/Setup/' + s_dirs['example_name'] + '_all_molecules' + '.xyz','w+') as input_file:
        print('%d \n' % num_gas_molecules, file=input_file)
        s1 = 'MET'
        for i in range(0, num_gas_molecules ):
            print(s1, x[i],' ', 
                      y[i],' ',
                      z[i],' ', file=input_file)



    with open(s_dirs['quick_route'] + '/Setup/' + s_dirs['example_name'] + '_all_molecules.lt','w+') as input_file: 
        print('import "methane.lt"', file=input_file)
        print('# Periodic boundary conditions:', file=input_file)
        print('part = new MET[',num_gas_molecules,']', file=input_file)
        print('write_once("Data Boundary") {', file=input_file)
        
        x_max = np.size(vol_3d,0)*mol_space
        y_max = np.size(vol_3d,1)*mol_space
        z_max = np.size(vol_3d,2)*mol_space
    
        print(0.0, x_max, 'xlo xhi', file=input_file)
        print(0.0, y_max, 'ylo yhi', file=input_file)
        print(0.0, z_max, 'zlo zhi', file=input_file)
        print( '}\n', file=input_file )
        
        
    plot_molecules(np.stack((x,y,z), axis=1),1)
    plt.savefig(s_dirs['quick_route'] + '/Setup/full_system.png')
    plt.close()
    

    return xs.shape[0]



def get_solid_coords(vol_3d, mol_space=1):
    
    """
    Reads an n-dim binary image (0,1) called vol_3d
    and returns the positions of the solids (1) 
    in a (points, n) numpy array spaced mol_space
    
    
    """
    
    indices = vol_3d.nonzero()
    coords  = np.stack(indices, axis=1)*mol_space + mol_space/2 
    return coords



def write_moltemplate_file(coords, vol_3d, mol_space=1, s_dirs='tmp'):
    """
    Writes a moltemplate input file that defines the positions of the solid
    structure.
    """

    
    with open(s_dirs['quick_route']  + '/Setup/' + s_dirs['example_name'] + '_solid.lt','w+') as input_file: 
        print('import "methane.lt"', file=input_file)
        print('# Periodic boundary conditions:', file=input_file)
        print('write_once("Data Boundary") {', file=input_file)
        
        x_max = np.size(vol_3d,0)*mol_space
        y_max = np.size(vol_3d,1)*mol_space
        z_max = np.size(vol_3d,2)*mol_space
    
        print(0.0, x_max, 'xlo xhi', file=input_file)
        print(0.0, y_max, 'ylo yhi', file=input_file)
        print(0.0, z_max, 'zlo zhi', file=input_file)
        print( '}\n', file=input_file )
        
        s1 = 'metane'
        s2 = ' = new MET.move('
        
        for i in range(0, np.size(coords,0) ):
            
            s3 = s1 + str(i + 1)
            print(s3,s2,coords[i,0],',',coords[i,1],',',coords[i,2],')', file=input_file)
           

def write_packmol_file(vol_3d, mol_space, num_molecules, s_dirs):
    """
    Writes a packmol input file that defines the positions of the gas
    """
    
    with open(s_dirs['quick_route'] + '/Setup/' + s_dirs['example_name']  + '_free_gas.inp','w+') as input_file: 
        print('tolerance 2.0', file=input_file)
        print('filetype xyz', file=input_file)
        print('output ' + s_dirs['example_name'] + '_free_gas.xyz', file=input_file)
        print('\n', file=input_file)
        print('structure methane.xyz', file=input_file)
        print('\tnumber %d' % num_molecules, file=input_file)
        
        
        x_max = np.size(vol_3d,0)*mol_space
        y_max = np.size(vol_3d,1)*mol_space
        z_max = np.size(vol_3d,2)*mol_space
        
        #print('\tinside box ' + '0.0 0.0 0.0' ,
        #       x_max, y_max, z_max, file=input_file)
        
        print('\tinside box ' , mol_space/2, mol_space/2, mol_space/2,
               x_max-mol_space/2, y_max-mol_space/2, z_max-mol_space/2, file=input_file)
                
        
        print( 'end structure', file=input_file )
        





def read_lammps_output(dir_path, file_type, d_factor=1):
    """
    Reads the output from lammps NVT
    
    Note: takes in account squared cross-section (YZ)
    
    """
    
    
    #s_dirs={}
    #s_dirs['quick_route'] = 'ellipse/ellipse_19_29_1_30_rho_200'
    
    if file_type == 'initial':
        df = pd.read_csv(dir_path + '/NVT/density_gas_first_ts.profile') #reads lammps 
    elif file_type == 'final':
        df = pd.read_csv(dir_path + '/NVT/density_gas_final.profile') #reads lammps 
    
    
    
    info = df['# Chunk-averaged data for fix 92 and group ave'][2].split() 
    num_chunks = int( info[1] ) #number of chunks x*y
    
    ts = int(df.shape[0]/num_chunks) #number of time steps
    
    X = df['# Chunk-averaged data for fix 92 and group ave'].str.split().to_numpy()

    lammps_out = np.zeros((num_chunks,3,ts))
    # Y_center, Z_center, Atom Count, Density [g/cc]
    
    ts_count   = 0
    for line in range(3, df.shape[0]):
        if np.shape( X[line] )[0] == 5:
            lammps_out[int( X[line][0] )-1,0,ts_count] = float(X[line][1]) #Y
            lammps_out[int( X[line][0] )-1,1,ts_count] = float(X[line][2]) #Z
            #lammps_out[int( X[line][0] )-1,2,ts_count] = float(X[line][3]) #Atoms
            lammps_out[int( X[line][0] )-1,2,ts_count] = float(X[line][4]) #Rho
        if np.shape( X[line] )[0] == 3:
            ts_count= ts_count + 1
            
            
    side_l = int(num_chunks**0.5) #side length (square cross-section)
    t_side = side_l//d_factor  #target side size (according to the user)
    lammps_2d = lammps_out.reshape( (side_l,side_l,3,ts) )
    
    downscaled_rho = lammps_2d[:,:,2,:].reshape( (t_side,side_l//t_side,
                                                  t_side,side_l//t_side,ts)
                                                   ).mean((-2,-4)) 
                                                    #averages 
                                                    #to get the desired size
                                                    
    downscaled_rho = downscaled_rho[:,:,-1]                                    
                                                    
    #plt.figure();plt.imshow(downscaled_rho[:,:]);plt.colorbar()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    yys,zzs = np.meshgrid(range(32),range(32))
#    ax.plot_surface(yys, zzs, lammps_out[:,3,0].reshape(32,32))
#    plt.show()
    
    
    return downscaled_rho
            

def plot_molecules(coords, plot_num=10):
    fig = plt.figure(plot_num)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( coords[:,0],coords[:,1],coords[:,2]) 
    plt.title('Methane lattice')
    ax.view_init(elev=15, azim=-25)

        
def read_min(dir_path, num_solid_mol):
    
    num_solid_mol = 7200
    
    #dir_path = 'ellipse/ellipse_19_29_1_30_rho_200'
    
    df = pd.read_csv(dir_path + '/Minimization/system_after_min.data') #reads lammps 
    X = df['LAMMPS data file via write_data'].str.split().to_numpy()
    num_atoms = int(X[0][0])
    
    lammps_out = np.zeros((num_atoms,3))
    for line in range(16, num_atoms+16):
        lammps_out[int(X[line][0])-1,0] = X[line][4] 
        lammps_out[int(X[line][0])-1,1] = X[line][5] 
        lammps_out[int(X[line][0])-1,2] = X[line][6] 
        
    
    
    #plot_molecules(lammps_out[num_solid_mol:-1,:],2)
    
    gas_coords = np.trunc( lammps_out[num_solid_mol:-1,:]/0.9325 ).astype(int)
    
    
    domain_3d = np.zeros( (int(3.73*100/.9325),
                           int(3.73*32/.9325),
                           int(3.73*32/.9325)) )
    
    
    for mols in gas_coords:
        domain_3d[mols[0],mols[1],mols[2]] += 1
    
    domain_3d = domain_3d*16.04
    
    ds = domain_3d.mean(axis=0).reshape( (32,4,32,4) ).mean((1,3))
    
    












##########################################################
##########################################################
##########    Old stuff (might be useful later on)   #####
##########           Proceed with caution            #####
##########################################################
##########################################################

#def old_get_solid_coordinates(vol_3d, mol_space=1):
#    """
#    Reads a binary image (0,1) named vol_3d
#    and returns the positions of the solids (1) accordingly
#    """
#    
#    x = np.linspace(0,np.size(vol_3d,0),np.size(vol_3d,0),endpoint=False)*mol_space
#    y = np.linspace(0,np.size(vol_3d,1),np.size(vol_3d,1),endpoint=False)*mol_space
#    z = np.linspace(0,np.size(vol_3d,2),np.size(vol_3d,2),endpoint=False)*mol_space
#    
#    
#    
#
#    # Each element of `range_dims`
#    # ranges over the dimension length for dim of x, y and z
#    range_dims = [ np.linspace(0,d,d,endpoint=False)*3.7
#                for d in vol_3d.shape]
#    
#
#    num_molecules   = (vol_3d==1).sum() # Count the number of ones in vol_3d
#    coords = np.zeros((num_molecules, 3))
#    
#    for i,mesh_dim_coordinates in enumerate(np.meshgrid(*range_dims, indexing='ij')):
#        coords[:,i] = mesh_dim_coordinates[vol_3d==1]
#        
#
#    return coords
#        
#
#            
#    
#
#    
#    
#def BU_get_solid_coordinates(vol_3d, mol_space=1):
#    """
#    Reads a binary image (0,1) named vol_3d
#    and returns the positions of the solids (1) accordingly
#    """
#    
#    x = np.linspace(0,np.size(vol_3d,0),np.size(vol_3d,0),endpoint=False)*mol_space
#    y = np.linspace(0,np.size(vol_3d,1),np.size(vol_3d,1),endpoint=False)*mol_space
#    z = np.linspace(0,np.size(vol_3d,2),np.size(vol_3d,2),endpoint=False)*mol_space
#    
#
#    
#    x, y, z = [ np.linspace(0,d,d,endpoint=False)*3.7
#                for d in vol_3d.shape]
#    
#    [X,Y,Z] = np.meshgrid(x,y,z, indexing='ij')
#    
#    #vol3d.astype(bool).sum()
#    
#    num_molecules   = vol_3d[vol_3d==1].size
#    coords = np.zeros((num_molecules, 3))
#    
##    for i,dim_coords in []
#    x_coords = X[vol_3d==1]
#    y_coords = Y[vol_3d==1]
#    z_coords = Z[vol_3d==1]
#    
#    coords[:,0] = x_coords
#    coords[:,1] = y_coords
#    coords[:,2] = z_coords
#    
#    return coords
#
#def old_write_moltemplate_file(coords, vol_3d, mol_space=1, file_name='tmp'):
#    """
#    Writes a moltemplate file that defines the positions of the solids
#    """
#    
#    #context manger with "with" statement:
#    # with expr as name:
#    # computes expr
#    # activates expr.__enter__()
#    # try: 
#    #         run code in block
#    #
#    # finally:
#    #        run expr.__exit__(error_state)
#    
#    
#    #try:
#    #    f = open(fname)
#    #    do stuff that might error
#    #except ValueError as ve:
#    #   do stuff if a value error occurred
#    #except TypeError....
#    #Except (AssertionError,FileNotFoundError) as something_else:
#    #     try to recover from the error
#    #else: #if no error occurred, do this
#    #
#    #finally: 
#    #    f.close()
#    #    #do stuff regardless of the error or no error
#    with open(file_name + '.lt','w+') as input_file: 
#        print('import "methane.lt"', file=input_file)
#        print('# Periodic boundary conditions:', file=input_file)
#        print('write_once("Data Boundary") {', file=input_file)
#        
#        x_max = np.size(vol_3d,0)*mol_space
#        y_max = np.size(vol_3d,1)*mol_space
#        z_max = np.size(vol_3d,2)*mol_space
#    
#        print(0.0, x_max, 'xlo xhi', file=input_file)
#        print(0.0, y_max, 'ylo yhi', file=input_file)
#        print(0.0, z_max, 'zlo zhi', file=input_file)
#        print( '}\n', file=input_file )
#        
#        
#        
#        s1 = 'metane'
#        s2 = ' = new MET.move('
#        
#        
#        
#        for i in range(0, np.size(coords,0) ):
#            
#            s3 = s1 + str(i)
#            print(s3,s2,coords[i,0],',',coords[i,1],',',coords[i,2],')', file=input_file)
#            #input_file.write()
#            
#            #input_file.writelines(l1)
#            
#            
#def OLDwrite_final_coordinates(vol_3d, file_name):
#    """
#    Reads the packmol output file, and erases the molecules overlapping with
#    the solid
#    Writes the input coordinates for lammps
#    
#    
#    ...FIX DESCRIPTION...
#    
#    
#    Comments: Might not be general enough for any 3D geometry
#    In simple cases, the central region is labeled as 2, this
#    might have to be assessed case-by-case
#    """
#             
#    # solid coordinates
#    *_,xs,ys,zs = np.loadtxt('Setup/' + file_name+'.data',skiprows=25,unpack=True)
#    # free gas coordinates
#    xg,yg,zg    = np.loadtxt('Setup/' + file_name + '_free_gas.xyz',
#                             usecols=(1,2,3), skiprows=2,unpack=True)/3.73
#    
#    
#    # Obtain the voxelized coordinates of the molecules
#    gas_indices = np.trunc( (xg, yg, zg) ).transpose().astype('int')
#    
#    
#    """
#    3D Plot
#    """
#    gas_coords = np.stack((xg,yg,zg), axis=1)
#    plot_molecules(gas_coords)
#    
#    
#    # find connected components
#    vol_3d_bis  = vol_3d + 1; vol_3d_bis[vol_3d_bis==2]=0 #changes 0's to 1's
#    filter_conn = np.ones((3,3,3), dtype=np.int) #filter to find connected regions
#    
#    
#    vol_3d_bis, ncomponents = find_conn_comps(vol_3d_bis, filter_conn)
#    print(ncomponents,'void connected regions where found.\n')
#    plt.imshow(vol_3d_bis[0,:,:])
#    
#    geometry_center_label = 2 #WARNING: This might not hold for every case
#                              #the main area of interested might be labed differently
#                              #works for now. Maybe we could compare the areas
#    
#    vol_3d_bis[ vol_3d_bis != geometry_center_label ] = 0
#    vol_3d_bis[ vol_3d_bis == geometry_center_label ] = 1
#    plt.figure(); plt.imshow(vol_3d_bis[0,:,:])
#    
#    #extract the indices where gas can be placed, and create a (n,3) coordinate array
#    void_indices = np.stack( vol_3d_bis.nonzero(), axis=1 ) 
#    
#    # find where the coordinates overlap (dis=0) and extract the coordinates that
#    # index the free gas molecule array. this returns the coordinates of the 
#    # gas inside the structure. The 'valid gas' molecules
#
#
#    
#    
#    """
#    Placeholder for Nick's approach:
#        vol_3d_bis[truncated_gas_coords] to create a mask
#        should work with big big arrays
#    """
#    
#    
#    valid_gas = {}
#    
#    for i in range( 0, gas_indices.shape[0] ):
#        print(gas_indices[i,0], gas_indices[i,1], gas_indices[i,2])
#        valid_gas[i] = vol_3d_bis[ gas_indices[i,0], gas_indices[i,1], gas_indices[i,2] ] 
#        
#        
#        
#        
#        
#
#
#    valid_gas = (distances(gas_indices,void_indices)==0 ).nonzero()[0]
#    xg_valid, yg_valid, zg_valid = xg[valid_gas], yg[valid_gas], zg[valid_gas]
#     
#    
#    
#    #Write all the solid and valid gas molecule coordinates to a file
#    x,y,z = np.concatenate((xs,xg_valid)), \
#            np.concatenate((ys,yg_valid)), \
#            np.concatenate((zs,zg_valid))
#            
#    num_gas_molecules = np.size(x)
#    
#    with open('Setup/' + file_name + '_allMolecules' + '.xyz','w+') as input_file:
#        print('%d \n' % num_gas_molecules, file=input_file)
#        s1 = 'MET'
#        for i in range(0, num_gas_molecules ):
#            print(s1, x[i],' ', 
#                      y[i],' ',
#                      z[i],' ', file=input_file)
           
        