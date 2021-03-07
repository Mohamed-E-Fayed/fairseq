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
    """ This class is a general class for using language model before and after main model"""

    def __init__(self, args, lm, encoder, decoder):
        self.lm1=FairseqEncoder(args, lm)
        self.main_model=transformer.TransformerModel(args, encoder, decoder)
        self.lm2=FairseqEncoder(args, lm)

    @staticmethod
    def add_args(parser):
        # specific model arguments
        parser.add_argument("--lm1-name", type=str, metavar='N',
                help="Name of language model no. 1.",)
        parser.add_argument("--lm1-layers", type=int, metavar='N',
                help="Number of layers of language model no. 1. It is converted into both --encoder-layers and --decoder-layers for language models only\n Then, the correct one is used by the corresponding model",)
        parser.add_argument("--lm1-dropout",
                help="probability dropout used in language model no. 1",)
        parser.add_argument("--lm2-name", type=str, metavar='N',
                help="Name of language model no. 2.",)
        parser.add_argument("--lm2-layers", type=int, metavar='N',
                help="Number of layers of language model no. 2. It is converted into both --encoder-layers and --decoder-layers for language models only\n Then, the correct one is used by the corresponding model",)
        parser.add_argument("--lm2-dropout",
                help="probability dropout used in language model no. 2",)
        parser.add_argument("--tokens-per-sample", type=int, metavar='N',
                help="number of tokens per sample. I do not understand its use yet",)
        parser.add_argument('--model-dropout',
            help="Dropout probability",
        )
        parser.add_argument("--min-lr",
            help="minimum allowed value for learning rate",
            default=0,
        )
        parser.add_argument("--model-encoder-layers", type=int, metavar='N',
            help="number of layers of main model encoder",
        )
        parser.add_argument("--model-encoder-embed-dim", type=int, metavar='N',
            help="embeddings dimension in main model encoder layers",
        )
        parser.add_argument("--model-encoder-attention-heads", type=int, metavar='N',
            help="number of attention heads in main model encoder",
        )
        parser.add_argument("--model-encoder-ffn-embed-dim", type=int, metavar='N',
            help="main model encoder embedding dimension for FFN",
        )
        parser.add_argument("--model-decoder-layers", type=int, metavar='N',
            help="number of layers of decoder of main model",
        )
        parser.add_argument("--model-decoder-embed-dim", type=int, metavar='N',
            help="embeddings dimension in main model decoder layers",
        )
        parser.add_argument("--model-decoder-attention-heads", type=int, metavar='N',
            help="number of attention heads in main model decoder",
        )
        parser.add_argument("--model-decoder-ffn-embed-dim", type=int, metavar='N',
            help="main model decoder embedding dimension for FFN",
        )

        parser.add_argument("--share-model-decoder-input-output-embed",
            help="share decoder input and output embeddings",
        )
        parser.add_argument("--share-all-embeddings", action='store_true',
        help="share main model encoder, decoder and output embeddings",
        )

    @classmethod
    def build_model(cls, args, task):
        lm1=cls.build_language_model(args, task)
        main_model=cls.build_main_model(args, task)
        main_model_enc, main_model_dec=main_model[0], main_model[1]
        lm2=cls.build_language_model(args, task)
        return (cls, args, lm1, main_model, lm2)


    @classmethod
    def build_main_model(cls, args, task):
        # This function should return the correct model regardless of it is transformer or not.
        # I am not sure whether it works properly or not
        #return transformer.TransformerModel.build_model(args, task)
        args.encoder_embed_path=args.decoder_embed_path=None
        encoder_embed_tokens=transformer.TransformerModel.build_embedding(args, task.source_dictionary, args.model_encoder_embed_dim, args.encoder_embed_path)
        decoder_embed_tokens=transformer.TransformerModel.build_embedding(args, task.target_dictionary, args.model_decoder_embed_dim, args.decoder_embed_path)
        enc=transformer.TransformerModel.build_encoder(args, task.source_dictionary, encoder_embed_tokens)
        dec=transformer.TransformerModel.build_decoder(args, task.target_dictionary, decoder_embed_tokens)
        return (enc, dec)

    @classmethod
    def build_language_model(cls, args, task):
        return roberta.model.RobertaModel.build_model(args, task)
        #return  transformer_lm.TransformerLanguageModel.build_model(args, task)

@register_model_architecture("lm_transformer_lm", "lm_transformer_lm")
def base_architecture(args):
    args.lm1_name=getattr(args, "lm1_name", None)
    args.lm2_name=getattr(args, "lm2_name", None)
    args.tokens_per_sample=args.model_encoder_embed_dim
    args.lm1_dropout=getattr(args, 'lm1_dropout', 0.1)
    args.lm2_dropout=getattr(args, 'lm2_dropout', 0.1)


def get_main_model_arguments(args):
    # The names we path in command line are different from what expected by each model,
    # so, I made this function to add the correct arguments required by main model.
    args.encoder_layers=args.model_encoder_layers
    args.encoder_embed_dim=args.model_encoder_embed_dim
    args.encoder_ffn_embed_dim=args.model_encoder_ffn_embed_dim
    args.encoder_attention_heads=args.model_encoder_attention_heads
        
    args.decoder_layers=args.model_decoder_layers
    args.decoder_embed_dim=args.model_decoder_embed_dim
    args.decoder_ffn_embed_dim=args.model_decoder_ffn_embed_dim
    args.decoder_attention_heads=args.model_decoder_attention_heads

    args.share_decoder_input_output_embed=args.share_model_decoder_input_output_embed

    args.dropout=args.model_dropout
    

def get_lm1_arguments(args):
    # The names we path in command line are different from what expected by each model,
    # so, I made this function to add the correct arguments required by Language Model no. 1
    args.encoder_layers=args.lm1_layers
    args.decoder_layers=args.lm1_layers
    args.dropout=args.lm1_dropout


def get_lm2_arguments(args):
    # The names we path in command line are different from what expected by each model,
    # so, I made this function to add the correct arguments required by Language Model no. 2
    args.encoder_layers=args.lm2_layers
    args.decoder_layers=args.lm2_layers
    args.dropout=args.lm2_dropout


