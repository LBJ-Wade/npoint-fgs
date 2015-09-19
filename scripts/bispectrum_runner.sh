#! /bin/bash
#$ -l h_rt=12:00:00
#$ -pe orte 100
#$ -cwd
#$ -V
#$ -t 1-4
#$ -o runs/
#$ -e runs/


export ilo=$[$[$SGE_TASK_ID - 1] * 25 + 1]
export ihi=$[$ilo + 24]

for i in `seq $ilo $ihi`;
do 
    python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_tlm.npy --alm2 sim${i}_planck_353GHz_elm.npy --alm3 sim${i}_planck_353GHz_blm.npy --filename b_T353_E353_B353_lmax100.npy --lmax 100
done
