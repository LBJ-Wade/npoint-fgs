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

for i in `seq $ilo $ihi`;
do 
    #mpirun python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_tlm.npy --alm2 sim${i}_planck_353GHz_elm.npy --alm3 sim${i}_planck_353GHz_blm.npy --filename sim${i}_b_T353_E353_B353_lmax100.npy --lmax 100
    #mpirun python calc_bispectrum.py --alm1 sim${i}_planck_143GHz_tlm.npy --alm2 sim${i}_planck_353GHz_elm.npy --alm3 sim${i}_planck_353GHz_blm.npy --filename sim${i}_b_T143_E353_B353_lmax100.npy --lmax 100

    mpirun python calc_bispectrum.py --alm1 sim${i}_planck_143GHz_tlm.npy --alm2 sim${i}_planck_143GHz_elm.npy --alm3 sim${i}_planck_143GHz_blm.npy --filename sim${i}_b_T143_E143_B143_lmax100.npy --lmax 100

    mpirun python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_tlm.npy --alm2 sim${i}_planck_143GHz_elm.npy --alm3 sim${i}_planck_143GHz_blm.npy --filename sim${i}_b_T353_E143_B143_lmax100.npy --lmax 100

    mpirun python calc_bispectrum.py --alm1 sim${i}_planck_353GHz_tlm.npy --alm2 sim${i}_planck_217GHz_elm.npy --alm3 sim${i}_planck_217GHz_blm.npy --filename sim${i}_b_T353_E217_B217_lmax100.npy --lmax 100

done
