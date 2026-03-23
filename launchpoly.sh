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
    echo
    echo "Launching test program."
    echo
    module load cuda/12.1
    SEED=$3
    make polydynOpt
    ./polydynOpt test $SEED
elif [[ "$MODE" == "slurm" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Insufficient number of arguments. Arguments must include the time for job run in the formah HH:MM:SS"
        exit 1
    elif ! [[ "$TIME" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
        echo "Error: time format must be HH:MM:SS"
        exit 1
    else
        echo
        echo "Submitting job to SLURM for time $2."
        echo
        sbatch --time="$TIME"  ./launchpolyslurm.sh
    fi

elif [[ "$MODE" == "slurmarray" ]]; then
    if [[ $# -ne 3 ]]; then
        echo "Insufficient number of arguments. Arguments must include the time for job run in the formah HH:MM:SS and the number of arrays"
        exit 1
    elif ! [[ "$TIME" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
        echo "Error: time format must be HH:MM:SS"
        exit 1
    else
        ARRAYS=$3
        echo
        echo "Submitting job to SLURM for time $2 with $3 arrays."
        echo
        sbatch --time="$TIME" --array=0-$ARRAYS  ./launchpolyslurm.sh
    fi
else
   echo "Missing mode of launching. Give as argument either 'test' or 'slurm'."
   exit 1
fi