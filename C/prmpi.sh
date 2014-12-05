#!/bin/bash
#SBATCH -J myMPI           # job name
#SBATCH -o myMPI.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 32              # total number of mpi tasks requested
#SBATCH -p development     # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
# SBATCH --mail-user=aaron.myers@utexas.edu
# SBATCH --mail-type=begin  # email me when the job starts
# SBATCH --mail-type=end    # email me when the job finishes
ibrun mpiload /scratch/01396/naga86/hw7/livejournal.dat .85 1 .8 .5          # run the MPI executable named a.out
