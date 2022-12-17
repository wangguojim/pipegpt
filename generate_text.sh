#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
CHECKPOINT_PATH=ckpts/megatron_4tp_7000
VOCAB_FILE=data/ch_vocab/vocab.txt
Special_Token_File=data/ch_vocab/special_tokens.yaml
b=1
mp=4
#pp=4
experts=2
nodes=1
gpus=4


use_tutel=""
#use_tutel="--use-tutel"


#ds_inference=""
ds_inference="--ds-inference"

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
L=24
H=2304
A=32
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --pipeline-model-parallel-size $pp \
       --num-layers $L \
       --hidden-size $H \
       --num-attention-heads $A \
       --max-position-embeddings 1024 \
       --tokenizer-type BertWordPieceCase \
       --fp16 \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 10 \
       --out-seq-length 10 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
#       --merge-file $MERGE_FILE \
       --special-token-file $Special_Token_File \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples $((100*$b))
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd
