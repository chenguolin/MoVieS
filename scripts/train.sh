NUM_MACHINES=1        # set your machine number here
NUM_LOCAL_GPUS=8      # set your GPU number per machine here
MACHINE_RANK=0        # set the rank of this machine here
MAIN_MACHINE_IP=""    # set your machine IP here
MAIN_MACHINE_PROT=""  # set your machine port here

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# export HF_ENDPOINT=https://hf-mirror.com
# export WANDB_BASE_URL=https://api.bandw.top
export HF_HOME=~/.cache/huggingface
export TORCH_HOME=~/.cache/torch
export NCCL_DEBUG=VERSION
export HDF5_USE_FILE_LOCKING=FALSE

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_MACHINE_IP \
    --main_process_port $MAIN_MACHINE_PROT \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
$@
