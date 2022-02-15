# cd /home/jesantos/.local/share/jupyter/runtime

from scipy.ndimage.morphology import distance_transform_edt as e_dist
from scipy.ndimage.measurements import label as find_conn_comps
import matplotlib.pyplot as plt
import pore_utils as pore #imports my library
from PIL import Image
import numpy as np
import cv2

sys_name = 'jesantos'

s_dirs_supercomp = {
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
if sys_name == 'supercomp'  : s_directories = s_dirs_supercomp
    

def trim_im(im):
    true_points = np.argwhere(im)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = im[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
              top_left[1]:bottom_right[1]+1]  # inclusive
    
    
    if out.shape[0]>28 or out.shape[1]>28:
        out = cv2.resize(out, (28,28), interpolation=cv2.INTER_AREA)
    else:
         im_c1 = int(out.shape[0]/2)*2
         im_c2 = int(out.shape[1]/2)*2
         out = cv2.resize(out, (im_c1,im_c2), interpolation=cv2.INTER_AREA)
        
    
    out_f = np.zeros((32,32))
    
    im_c1 = int(out.shape[0]/2)
    im_c2 = int(out.shape[1]/2)
    
    out_f[16-im_c1:16+im_c1,16-im_c2:16+im_c2] = (out>0.5)*1
    
    return out_f



shale = Image.open('files/shale_first_slice.png')
shale = np.array(shale)
shale=(shale==0).astype(int)
pores, _ = find_conn_comps(shale)

pore_num = [659, 662, 642, 50, 261, 207, 179, 782, 626, 362, 88, 780, 750, 258, 
            595, 293, 85, 566, 573, 405]

pore_num_bis = [449, 571, 381, 720, 767, 293, 88, 209, 739, 467, 236]



pore_num = np.random.randint(0,799,size=700)



ind_pores = np.zeros( (len(pore_num),32,32) )

for i in range(len(pore_num)):
    if np.sum(( pores==pore_num[i] )*1.0) > 20:
        ind_pores[i, :, :] = trim_im( ( pores==pore_num[i] )*1.0 )
    
    

#gas_density = 50
y_molecules = 32 #Cross section axis 1
z_molecules = 32 #Cross section axis 2
tube_length = 100 # num of molecules on the periodic dir
molecular_spacing = 3.73 #for methane
molar_mass = 16.04 #[g/mol]
NVT_ts     = 500010




for i in range(0,500):
    
    gas_density = np.random.randint(50,200)
    #gas_density = 250
    
    #vol_3d = ind_pores[np.newaxis,i,:,:]
    vol_3d = ind_pores[np.newaxis,i,:,:]
    
    
    if np.sum(vol_3d)<20:
        continue
    
    vol_3d = vol_3d+1
    vol_3d[vol_3d==2]=0
    vol_3d = e_dist(vol_3d)
    vol_3d[vol_3d!=1] = 0
    vol_3d = np.repeat(vol_3d, tube_length, axis=0)
    coords = pore.get_solid_coords(vol_3d, molecular_spacing)
    s_directories['quick_route'] = ('vaca_muerta_random/pore_%2.2d_%3.3d' 
                                     % (i+500,gas_density) )
    s_directories['example_name'] = str(i)
    pore.create_folders(s_directories)
    pore.write_moltemplate_file(coords, vol_3d, molecular_spacing, s_directories)
    pore.run_moltemplate(s_directories, op='solid', loc='local')
    
    num_molecules = pore.compute_num_molecules(gas_density, molar_mass, 
                                               molecular_spacing, vol_3d)
                                  
    pore.write_packmol_file(vol_3d, molecular_spacing, num_molecules, 
                            s_directories)

    pore.run_packmol(s_directories)
    
    
    num_solid_mol = pore.write_final_coordinates(vol_3d, molecular_spacing,
                                                 s_directories,
                                                 cs_shape=None)
    
    pore.run_moltemplate(s_directories, 'all', loc='local')
    
    pore.create_Minimization_files(s_directories, num_solid_mol, bcs='p p p')
    
    pore.run_lammps('Minimization', s_directories)
    
    pore.create_NVT_files(s_directories, 1, num_solid_mol, 'first_ts')
    pore.run_lammps('NVT', s_directories)
    pore.create_NVT_files(s_directories, NVT_ts, num_solid_mol, 'normal')
    pore.run_lammps('NVT', s_directories)

