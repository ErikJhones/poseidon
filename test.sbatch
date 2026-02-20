#!/bin/bash

#SBATCH --partition=a6000
#SBATCH --job-name=tabicl_test    # Job name
#SBATCH --output=job.%x.%j.out # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --time=00:01:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-debug         # QOS to be used

echo "==============================="
echo "Job started"
echo "JobID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==============================="

# Start timer
START_TIME=$(date +%s)

# Rodar código Python
python3 test.py

# End timer
END_TIME=$(date +%s)

ELAPSED=$((END_TIME-START_TIME))

echo "==============================="
echo "JOB FINISHED"
echo "==============================="

echo "Elapsed time (seconds): $ELAPSED"
echo "Elapsed time (minutes): $(($ELAPSED/60))"

echo ""
echo "===== SLURM RESOURCE USAGE ====="

sacct -j $SLURM_JOB_ID \
--format=JobID,State,Elapsed,MaxRSS,AllocCPUS

echo "==============================="