# Scripts to perform MD simulations in pores of predetermined shapes. 
The create_trainingset.py script contains instructions to create random cross-sections with random gas density and  wrapper to run 3d lammps NVT simulations

## Instructions
1) Install the required dependencies by: pip install --user --requirement requirements.txt
2) Select your systems' name
3) Modify the 's_directories' dictionary to reflect where the executables are in your machine
4) load the adequate MPI compiler (ie: module load mpich/3.2.1-intel_17.0.6 )
5) Specify how many geometries are requested (num_cases)
6) Run the code: python create_trainingset.py
