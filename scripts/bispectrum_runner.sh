#! /bin/bash
#$ -l h_rt=12:00:00
#$ -pe orte 25
#$ -cwd
#$ -V
#$ -t 1-20
#$ -o runs/
#$ -e runs/


export ilo=$[$[$SGE_TASK_ID - 1] * 5 + 1]
export ihi=$[$ilo + 4]
export lmax=200

for i in `seq $ilo $ihi`;
do 
    mpirun python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_lmax${lmax}_Tlm.npy --alm2 sim${i}_planck_353GHz_lmax${lmax}_Elm.npy --alm3 sim${i}_planck_353GHz_lmax${lmax}_Blm.npy --filename sim${i}_T353_E353_B353_lmax${lmax}.npy --lmax $lmax
    
done
