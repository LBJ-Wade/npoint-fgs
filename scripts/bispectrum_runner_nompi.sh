#! /bin/bash
#$ -l h_rt=12:00:00
#$ -cwd
#$ -V
#$ -t 1-100
#$ -o runs/
#$ -e runs/

export i=$SGE_TASK_ID
export lmax=200

mpirun python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_lmax${lmax}_Tlm.npy --alm2 sim${i}_planck_353GHz_lmax${lmax}_Elm.npy --alm3 sim${i}_planck_353GHz_lmax${lmax}_Blm.npy --filename sim${i}_T353_E353_B353_lmax${lmax}.npy --lmax $lmax
    

