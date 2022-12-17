template_json="./examples/MoE/ds_config_gpt_TEMPLATE-Copy1.json"
GLOBAL_BATCH_SIZE=1
BATCH_SIZE=1
LOG_INTERVAL=1
CL_ENABLED=1
CL_START_SEQLEN=1
SEQ_LEN=1
CL_STEP=1
TENSORBOARD_DIR="tmp"

sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
    | sed "s/TENSORBOARD_DIR/\"${TENSORBOARD_DIR}\"/" \
          >tmp.json
