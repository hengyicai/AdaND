#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""
import locale
import os

from parlai.scripts.train_model import setup_args, TrainLoop

locale.setlocale(locale.LC_ALL, 'en_US')
PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "datatype": 'train',
    "batchsize": 128,
    "learningrate": 0.001,
    "dropout": 0.1,
    "gradient_clip": 0.1,
    "batch_sort": True,
    "validation_every_n_secs": -1,
    "validation_every_n_epochs": 0.5,
    "validation_metric": 'ppl',
    "validation_metric_mode": 'min',
    "validation_patience": 10,
    "log_every_n_secs": 1,
    "shuffle": True,
    "numworkers": 40,
    "multigpu": False,
    "num_epochs": -1,
    "history_size": -1,
    "truncate": 50,
    "label_truncate": 50,
    "text_truncate": 50,
    "beam_size": 1,
    "input_dropout": 0.1,
    "label_smoothing": 0.0,
    "eval_embedding_type": os.path.join(
        PARLAI_HOME, 'data/adand_data/eval_embedding.vec'
    ),
    "max_train_time": -1,
}

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='AdaND_Data',
        model='parlai.agents.AdaND.AdaND:AdaNDAgent',
        model_file=os.path.join(
            PARLAI_HOME, 'models/AdaND/AdaND'
        ),
        dict_lower=True,
        dict_minfreq=-1,
        dict_maxtokens=20000,
        hiddensize=300,
        embeddingsize=300,
        attention='general',
        attention_time='post',
        numlayers=2,
        rnn_class='context_topic_adaptive_lstm',
        lookuptable='unique',
        optimizer='adam',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=True,
        numsoftmax=1,
        weight_decay=3e-7,
        show_advanced_args=True,
        adaptive_input_size=32,
        adaptive_hidden_size=128,
        num_topics=5,
        topic_dict=os.path.join(
            PARLAI_HOME, 'data/adand_data/topic_vocab.dict'
        ),
        latent_size=64,
        bow_hiddensizes="512, 128, 64",
        ensemble_factors=128,
    )
    parser.set_defaults(**OVERRIDE)
    opt = parser.parse_args()

    opt['override'] = opt['override'] if 'override' in opt else {}
    # Add arguments of OVERRIDE into opt or opt['override']
    for k, v in OVERRIDE.items():
        if k not in opt:
            print("[ Add {} with value {} to opt ]".format(k, v))
            opt[k] = v
        else:
            # Add {k: opt[k]} to opt['override']
            opt['override'][k] = opt[k]
            print("[ Add {} with value {} to opt['override'] ]".format(k, opt[k]))

    TrainLoop(parser).train()
