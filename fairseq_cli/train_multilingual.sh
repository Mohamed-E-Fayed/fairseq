#!/usr/bin/env bash
#nvidia-smi

python3 -c "import torch; print(torch.__version__)"

src=$1
tgt=$2
ARCH=lm_transformer_lm
MODELSIZE=transformer_big
DATAPATH=~/data/${src}-${tgt}
SAVEDIR=checkpoints_${ARCH}_$src_$tgt

mkdir -p $SAVEDIR
if [ ! -f "$SAVEDIR/checkpoint_last.pt" ]
then
warmup="--reset-lr-scheduler"
else
warmup=""
fi

fairseq-train $DATAPATH   \
	--task multilingual_translation  --lang-pairs fr-ar,es-ar,pt-ar,it-ar,ro-ar  \
  -a $ARCH --optimizer adam --lr 0.0005 \
--adam-betas '(0.9,0.98)' --save-dir $SAVEDIR $warmup \
--ddp-backend=no_c10d \
--dropout 0.3 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --max-update 15000000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
--clip-norm 0.1 --label-smoothing 0.1   \
--update-freq 4    --max-epoch 5  \
  --max-tokens 4096 --num-workers 8  --memory-efficient-fp16  --fp16-scale-tolerance=0.25  --min-loss-scale=1   \
--skip-invalid-size-inputs-valid-test \
--decoder-layers  6 --encoder-layers  6  \
--encoder-embed-dim 1024 --decoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096  --encoder-ffn-embed-dim 4096 \
--encoder-attention-heads 16 --decoder-attention-heads 16 \
  --share-decoders   --share-decoder-input-output-embed \
--keep-last-epochs 1 \
 | tee -a $SAVEDIR/training.log

