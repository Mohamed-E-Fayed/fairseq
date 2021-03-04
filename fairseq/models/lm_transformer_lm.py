# This file is copied from fairseq/models/transformer.py
# and edit in it
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    roberta,
    bart,
    transformer,
    transformer_lm,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
#from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model("lm_transformer_lm")
class LMTransformerLMModel(BaseFairseqModel):
    def __init__(self, args, lm, encoder, decoder):
        self.lm1=FairseqEncoder(args, lm)
        self.transformer=transformer.TransformerModel(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
	    cls_lm1, encoder_lm1=self.lm1.build_model(cls, args, task)
	    cls_model, args_model, encoder_model, decoder_model= self.transformer.build_model(cls, args, task)
	    cls_lm2, encoder_lm2=self.lm2.build_model(cls, args, task)
	    return (cls, args, encoder_lm1, encoder_model, decoder_model, encoder_lm2)

@register_model_architecture("lm_transformer_lm", "lm_transformer_lm")
def base_architecture(args):
		args.src_lm=getattr(args, "src_lm", None)
		args.tgt_lm=getattr(args, "tgt_lm", None)
