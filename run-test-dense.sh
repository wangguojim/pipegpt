#!/bin/bash

DIR=`pwd`

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

#mkdir -p $DIR/logs  # 创建logs目录
 


#DATASET_1="<PATH TO THE FIRST DATASET>"
#DATASET_2="<PATH TO THE SECOND DATASET>"
#DATASET_3="<PATH TO THE THIRD DATASET>"
#DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"
#DATASET=`python get_data_path.py`

DATASET="/raid/nlp/data/aggregated_meta_data/dataset_10b_22_text_document"

BASE_DATA_PATH=.  
#DATASET="1 /raid/nlp/processed_data/zhiyuan/part-2021278643_content_document 1 /raid/nlp/processed_data/THUCNews/股票/647234_text_document"
VOCAB_PATH=${BASE_DATA_PATH}/ch_tokenizer_data/vocab.txt
SPECIAL_TOKEN_PATH=${BASE_DATA_PATH}/ch_tokenizer_data/special_tokens.yaml

script_path=$(realpath $0)
# echo $script_path

script_dir=$(dirname $script_path)

CONFIG_JSON="$script_dir/ds_config.json"

USE_DEEPSPEED=1
ZERO_STAGE=0


# Debug
#TP=4
#PP=4
#LAYERS=8
#HIDDEN=512
#SEQ=1024
#GLOBAL_BATCH=128
#WORKER_STR="-i worker-0"


# 1.7B
TP=1
PP=8
HIDDEN=4096
LAYERS=32
NUM_ATTENTION_HEADS=32
SEQ=1024
GLOBAL_BATCH=960
WORKER_STR=""

MICRO_BATCH=10

CHECKPOINT_PATH=/raid/nlp/ckpts/ckpts_10b

while [ $# -gt  0 ]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ \
    --loss-scale 12 \
    --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters 10000 \
    --lr 1.0e-5 \
	--min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--special-token-file ${SPECIAL_TOKEN_PATH} \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
	--save-interval 100000 \
    --split 800,100,100 \
    --clip-grad 1.0 \
	--weight-decay 0.1 \
	--optimizer  onebit-adam\
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
	--tensorboard-dir $CHECKPOINT_PATH/tensorboard_dir \
    --fp16 \
	--checkpoint-activations
        "
#echo $USE_DEEPSPEED

if [ ${USE_DEEPSPEED} -eq 1 ]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"
fi
#echo $options
#exit

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "zero_optimization": {
        "stage": 0
    },
    "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 4e-4,
      "freeze_step": 2,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
    },

  "gradient_clipping": 1.0,
  "prescale_gradients": true,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

#run_cmd="deepspeed -i worker-0:0,1,2,3 ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i worker-0 ${DIR}/pretrain_gpt.py $@ ${options}"
run_cmd="deepspeed --hostfile hostfile ${DIR}/pretrain_gpt.py $@ ${options}"
#run_cmd="deepspeed -i localhost ${DIR}/pretrain_gpt.py $@ ${options}"


echo ${run_cmd}
eval ${run_cmd}
#eval nohup ${run_cmd} &

set +x


