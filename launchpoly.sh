#!/bin/bash

#Mode of execution
MODE=$1
#Time of execution if present
TIME=$2

set -e

#Move to program folder
cd /data/others/ciarchi/PolymerDyn/CudaPoly/
module load cuda/12.1

if [[ "$MODE" == "test" ]]; then
    echo "Launching test program."
    module load cuda/12.1
    make polydyn
    ./polydyn test
elif [[ "$MODE" == "slurm" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Insufficient number of arguments. Arguments must include the time for job run in the formah HH:MM:SS"
        exit 1
    elif ! [[ "$TIME" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
        echo "Error: time format must be HH:MM:SS"
        exit 1
    else
        echo "Submitting job to SLURM for time $2."
        sbatch --time="$TIME"  ./launchpolyslurm.sh
    fi
else
   echo "Missing mode of launching. Give as argument either 'test' or 'slurm'."
   exit 1
fi