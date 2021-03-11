#!/usr/bin/env bash
nvidia-smi

python3 -c "import torch; print(torch.__version__)"

src=$1
tgt=$2
ARCH=transformer_wmt_en_de_big_t2t
MODELSIZE=transformer_big
DATAPATH=~/data
SAVEDIR=checkpoints_3_not_segmented_${src}_${tgt}

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
--dropout 0.3 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 15000000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--clip-norm 0.1 --label-smoothing 0.1   \
--update-freq 4    --max-epoch 5  \
  --max-tokens 2048   --num-workers 8  --memory-efficient-fp16  --fp16-scale-tolerance=0.25  --min-loss-scale=1   \
--skip-invalid-size-inputs-valid-test \
--decoder-layers  12 --encoder-layers  12 \
--encoder-embed-dim 1024 --decoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096  --encoder-ffn-embed-dim 4096 \
--encoder-attention-heads 16 --decoder-attention-heads 16 \
  --share-decoder-input-output-embed \
  --eval-bleu \
					    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
					        --eval-bleu-detok moses \
						    --eval-bleu-remove-bpe \
							    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--keep-last-epochs 1 \
 | tee -a $SAVEDIR/training.log

