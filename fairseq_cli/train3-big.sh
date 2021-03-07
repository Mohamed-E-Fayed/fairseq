#!/usr/bin/env bash
#nvidia-smi

python3 -c "import torch; print(torch.__version__)"

src=$1
tgt=$2
ARCH=lm_transformer_lm
#ARCH=transformer
MODELSIZE=transformer_big
DATAPATH=~/data/${src}-${tgt}
SAVEDIR=checkpoints_${ARCH}_${src}_${tgt}

mkdir -p $SAVEDIR
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--reset-lr-scheduler"
else
warmup=""
fi

python3 train.py $DATAPATH    --cpu \
  -a $ARCH --optimizer adam --lr 0.0005 -s $src -t $tgt \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR $warmup \
--model-dropout 0.3 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 15000000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--clip-norm 0.1 --label-smoothing 0.1  --validate-interval-updates 15000 --patience 3  \
--keep-interval-updates 1 \
--update-freq 4    --max-epoch 5  \
  --max-tokens 2048 --num-workers 8  --memory-efficient-fp16  --fp16-scale-tolerance=0.25  --min-loss-scale=1   \
--skip-invalid-size-inputs-valid-test \
--model-decoder-layers  12 --model-encoder-layers  12 \
--model-encoder-embed-dim 1024 --model-decoder-embed-dim 1024 \
--model-encoder-ffn-embed-dim 4096  --model-encoder-ffn-embed-dim 4096 \
--model-encoder-attention-heads 16 --model-decoder-attention-heads 16 \
  --share-model-decoder-input-output-embed True \
  --eval-bleu \
					    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
					        --eval-bleu-detok moses \
						    --eval-bleu-remove-bpe \
							    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--keep-last-epochs 1 \
 | tee -a $SAVEDIR/training.log

