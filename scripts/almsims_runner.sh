#! /bin/bash
#$ -l h_rt=12:00:00
#$ -cwd
#$ -t 1-100
#$ -V
#$ -o runs/
#$ -e runs/

python simulate_alms.py sim$SGE_TASK_ID
