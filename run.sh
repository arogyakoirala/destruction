#!/bin/bash
EXERCISE_ENVIRONMENT="env"
source /lustre/ific.uv.es/ml/iae091/env/bin/activate
# eval "$(conda shell.bash hook)"
# conda activate $EXERCISE_ENVIRONMENT
chmod +x /lustre/ific.uv.es/ml/iae091/destruction/prep_data.sh
/lustre/ific.uv.es/ml/iae091/destruction/prep_data.sh /lustre/ific.uv.es/ml/iae091/