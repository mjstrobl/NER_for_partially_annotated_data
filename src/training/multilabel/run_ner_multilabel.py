#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

from datasets_multilabel import conll_file_to_features,get_labels, convert_examples_to_features, read_examples_from_file_whole_lines,convert_wiki_gold_features
import numpy as np
from transformers import BertForTokenClassification
from transformers import AutoConfig,AutoTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
import logging
import os
import random
import torch
import json
from seq_trainer_multilabel import train,evaluate



logger = logging.getLogger(__name__)


def main():
    config = json.load(open("../config/config.json"))

    model_name_or_path = config['model_name_or_path']    #model_name_or_path = "/media/michi/Data/models/seq_ner/english_conll_bert_base_cased_epochs_multilabel_2/checkpoint-10500/"
    cache_dir = None
    task_name = "ner"
    max_seq_length = 128
    data_dir = config['data_dir']
    model_type = "bert"

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"


    do_train = True
    do_eval = True
    do_predict = True
    do_wiki = True

    max_steps = -1
    save_steps = 750
    evaluate_during_training = True
    per_gpu_train_batch_size = 32
    gradient_accumulation_steps = 1
    num_train_epochs = 10
    num_train = -1
    weight_decay = 0.0
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    max_grad_norm = 1.0
    logging_steps = 2759

    output_dir = config['output_dir'] + 'multilabel/'

    label_list = get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    bert_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True
    )

    model = BertForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=bert_config,
        cache_dir=cache_dir
    )

    logger.info("Creating features from dataset file at %s", data_dir)
    modes = {"test","dev","train"}
    datasets = {}
    '''important_articles_lower = json.load(open(config['important_articles_lower']))


    wiki_features = conll_file_to_features(config['wiki_food'],max_seq_length,label_list, tokenizer)
    random.seed(10)
    random.shuffle(wiki_features)

    n_train = int(len(wiki_features) * 0.8)
    n_dev = int((len(wiki_features) - n_train) / 2)
    datasets['train'] = wiki_features[:n_train]
    datasets['dev'] = wiki_features[n_train:n_train+n_dev]
    datasets['test'] = wiki_features[n_train + n_dev:]
    print("Train on " + str(n_train) + " additional features from Wikipedia.")
    print("Test/Evaluate on " + str(n_dev) + " additional features from Wikipedia.")

    features = convert_wiki_gold_features(config['wiki_gold_filename'], label_list, max_seq_length, tokenizer)

    print("found " + str(len(features)) + " gold features.")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_weights = torch.tensor([f.weights for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_tokens = [f.tokens for f in features]

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_weights, all_label_ids)
    datasets['gold'] = dataset
    datasets['gold_tokens'] = all_tokens
'''
    
    for mode in modes:
        if not do_train and mode == "train":
            continue

        if not do_eval and mode == "dev":
            continue

        if not do_predict and mode == "test":
            continue

        examples = read_examples_from_file_whole_lines(data_dir, mode)
        features = convert_examples_to_features(examples,important_articles_lower,
                                                label_list,
                                                max_seq_length,
                                                tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_weights = torch.tensor([f.weights for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_tokens_conll = [f.tokens for f in features]

        dataset_conll = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_weights, all_label_ids)


        '''all_input_ids = torch.tensor([f.input_ids for f in datasets[mode]], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in datasets[mode]], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in datasets[mode]], dtype=torch.long)
        all_weights = torch.tensor([f.weights for f in datasets[mode]], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in datasets[mode]], dtype=torch.long)
        all_tokens_wiki = [f.tokens for f in datasets[mode]]

        dataset_wiki = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_weights, all_label_ids)

        datasets[mode + "_wiki"] = dataset_wiki
        datasets[mode + "_wiki_tokens"] = all_tokens_wiki

        features.extend(datasets[mode])

        if mode == "train" and num_train > -1 and len(features) > num_train:
            features = features[:num_train]


        all_tokens = []
        all_tokens.extend(all_tokens_conll)
        all_tokens.extend(all_tokens_wiki)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_weights = torch.tensor([f.weights for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        '''
        datasets[mode + "_conll"] = dataset_conll
        datasets[mode + "_conll_tokens"] = all_tokens_conll
        #datasets[mode] = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_weights, all_label_ids)
        datasets[mode + "_tokens"] = all_tokens

    # Training
    if do_train:

        model.to(device)

        train(config['results_dir'],output_dir,
              max_steps,
              save_steps,
              evaluate_during_training,
              datasets,
              model,
              tokenizer,
              label_list,
              num_labels,
              per_gpu_train_batch_size,
              gradient_accumulation_steps,
              num_train_epochs,
              weight_decay,
              learning_rate,
              adam_epsilon,
              warmup_steps,
              model_name_or_path,
              device,
              model_type,
              max_grad_norm,
              logging_steps)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model checkpoint to " + output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForTokenClassification.from_pretrained(output_dir)
        model.to(device)
    else:
        model = model.from_pretrained(output_dir)

    model.to(device)
    # Evaluation
    results = {}

    if do_eval:
        print("*** Evaluate ***")

        print("ALL:")
        evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='dev')

        print("CONLL:")
        evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='dev_conll')

        print("WIKI:")
        #evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='dev_wiki')

    # Predict
    if do_predict:
        print("*** Predict ***")
        print("ALL:")
        evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='test')

        print("CONLL:")
        evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='test_conll')

        print("WIKI:")
        #evaluate(config['results_dir'],datasets, model, per_gpu_train_batch_size, device, model_type, label_list,prefix='test_wiki')

    # Wiki
    if do_wiki:
        print("*** Predict ***")

        #evaluate(config['results_dir'], datasets, model, per_gpu_train_batch_size, device, model_type, label_list, prefix='gold')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

