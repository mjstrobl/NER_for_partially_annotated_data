# coding=utf-8
#
#   Modified by Michael Strobl 2022
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import logging
import os
import re
import json
import numpy as np

ENTITY_LENGTH = 6

logger = logging.getLogger(__name__)

RE_LINKS = re.compile(r'\[{2}(.*?)\]{2}', re.DOTALL | re.UNICODE)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tokens = tokens


def create_sentence(words, labels):
    sentence = ""
    entities = []
    current_start = -1
    current_end = -1
    current_label = None
    for i in range(len(words)):
        word = words[i]
        label = labels[i]

        if label[0] == 'B':
            if current_start > -1:
                entities.append((current_start, current_end, current_label))
            current_start = len(sentence)
            current_label = label[2:]
            sentence = sentence + word + ' '
            current_end = len(sentence) - 1
        elif label[0] == 'I':
            sentence = sentence + word + ' '
            current_end = len(sentence) - 1
        else:
            sentence = sentence + word + ' '
            if current_start > -1:
                entities.append((current_start, current_end, current_label))
            current_start = -1
            current_end = -1

    if current_start > -1:
        entities.append((current_start, current_end, current_label))

    entities.sort(key=lambda x: x[0], reverse=True)

    for tuple in entities:
        sentence = sentence[:tuple[0]] + '[[' + sentence[tuple[0]:tuple[1]] + '|' + sentence[tuple[0]:tuple[1]] + '|' + \
                   tuple[2] + ']]' + sentence[tuple[1]:]
    return sentence


def read_examples_from_file_whole_lines(data_dir, mode):
    #file_path = os.path.join(data_dir, "{}.txt".format(mode))
    file_path = data_dir + mode + ".txt"
    logger.info("Read file: %s", file_path)

    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    sentence = create_sentence(words, labels)
                    examples.append(sentence)
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    #labels.append(splits[-1].replace("\n", "")[0] + '-ENTITY')
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            sentence = create_sentence(words, labels)
            examples.append(sentence)

    return examples

def convert_examples_to_features(
        sentences,
        label_list,
        max_seq_length,
        tokenizer, pad_token_label_id=-1
):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for sentence in sentences:
        sentence = sentence.strip()
        feature = convert_line_to_feature(sentence, max_seq_length, label_map,
                                             tokenizer,pad_token_label_id=pad_token_label_id)
        features.append(feature)

    return features


def convert_line_to_feature(line, max_seq_length, label_map, tokenizer,
                            cls_token_at_end=False,
                            cls_token="[CLS]",
                            cls_token_segment_id=1,
                            sep_token="[SEP]",
                            sep_token_extra=False,
                            pad_on_left=False,
                            pad_token_label_id=-1,
                            pad_token=0,
                            pad_token_segment_id=0,
                            sequence_a_segment_id=0,
                            mask_padding_with_zero=True):
    tokens = []
    label_ids = []
    more_entities = 0

    while True:
        match = re.search(RE_LINKS, line)
        if match:
            entity = match.group(1)
            start = match.span()[0]
            end = match.span()[1]
            parts = entity.split('|')
            if len(parts) != 3:
                line = line[:start] + entity + line[end:]
                continue
            entity = parts[0]
            mention = parts[1]
            label = parts[2]
            before = line[:start]
            line = line[end:]

            word_tokens = tokenizer.tokenize(before)
            for i in range(len(word_tokens)):
                word_token = word_tokens[i]
                tokens.append(word_token)
                if word_token.startswith('#'):
                    label_ids.append(pad_token_label_id)
                else:
                    label_ids.append(label_map['O'])

            word_tokens = tokenizer.tokenize(mention)

            label = 'B-' + label
            for word_token in word_tokens:
                tokens.append(word_token)
                if word_token.startswith('#'):
                    label_ids.append(pad_token_label_id)
                else:
                    label_ids.append(label_map[label])
                label = 'I' + label[1:]
        else:
            break

    word_tokens = tokenizer.tokenize(line)
    for i in range(len(word_tokens)):
        word_token = word_tokens[i]
        tokens.append(word_token)
        if word_token.startswith('#'):
            label_ids.append(pad_token_label_id)
        else:
            label_ids.append(label_map['O'])


    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]


    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids.append(pad_token_label_id)

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids.append(pad_token_label_id)
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids.append(pad_token_label_id)
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,label_ids=label_ids, tokens=tokens)

def create_feature(tokens,label_ids,max_seq_length,tokenizer,
                            cls_token_at_end=False,
                            cls_token="[CLS]",
                            cls_token_segment_id=1,
                            sep_token="[SEP]",
                            sep_token_extra=False,
                            pad_on_left=False,
                            pad_token_label_id=-1,
                            pad_token=0,
                            pad_token_segment_id=0,
                            sequence_a_segment_id=0,
                            mask_padding_with_zero=True):
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids.append(pad_token_label_id)

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids.append(pad_token_label_id)
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids.append(pad_token_label_id)
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids,
                         tokens=tokens)

def conll_file_to_features(filename,max_seq_length, label_list, tokenizer,pad_token_label_id=-1):

    label_map = {label: i for i, label in enumerate(label_list)}

    tokens = []
    label_ids = []
    features = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                if len(tokens) > 0:
                    features.append(create_feature(tokens, label_ids, max_seq_length,tokenizer,pad_token_label_id=pad_token_label_id))
                    tokens = []
                    label_ids = []
            else:

                parts = line.split()
                word = parts[0]
                label = parts[1]
                word_tokens = tokenizer.tokenize(word)

                if label[0] == 'B' or label[0] == 'I':
                    current_label = label
                    for word_token in word_tokens:
                        tokens.append(word_token)
                        if word_token.startswith('#'):
                            label_ids.append(pad_token_label_id)
                        else:
                            label_ids.append(label_map[current_label])
                        current_label = 'I' + current_label[1:]
                else:
                    current_label = 'O'
                    for word_token in word_tokens:
                        tokens.append(word_token)
                        if word_token.startswith('#'):
                            label_ids.append(pad_token_label_id)
                        else:
                            label_ids.append(label_map[current_label])

    if len(tokens) > 0:
        features.append(create_feature(tokens, label_ids, max_seq_length,tokenizer,pad_token_label_id=pad_token_label_id))

    return features

def convert_wiki_gold_features(
        wiki_gold_filename,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_label_id=-1,
        pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
):
    #important_articles_lower = create_important_articles()
    #print('found ' + str(len(important_articles_lower)) + " important food articles.")

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    filepath = wiki_gold_filename
    with open(filepath) as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            tokens = []
            label_ids = []
            while True:
                match = re.search(RE_LINKS, line)
                if match:
                    entity = match.group(1)
                    start = match.span()[0]
                    end = match.span()[1]
                    parts = entity.split('|')

                    mention = parts[0]
                    label = parts[1]
                    before = line[:start]
                    line = line[end:]

                    word_tokens = tokenizer.tokenize(before)
                    tokens.extend(word_tokens)

                    for i in range(len(word_tokens)):
                        word_token = word_tokens[i]
                        if word_token.startswith('#'):
                            label_ids.append(pad_token_label_id)
                        else:
                            label_ids.append(label_map['O'])

                    word_tokens = tokenizer.tokenize(mention)

                    label = 'B-' + label
                    for word_token in word_tokens:
                        tokens.append(word_token)
                        if word_token.startswith('#'):
                            label_ids.append(pad_token_label_id)
                        else:
                            label_ids.append(label_map[label])
                            label = 'I' + label[1:]
                else:
                    break

            word_tokens = tokenizer.tokenize(line)
            tokens.extend(word_tokens)

            for i in range(len(word_tokens)):
                word_token = word_tokens[i]
                if word_token.startswith('#'):
                    label_ids.append(pad_token_label_id)
                else:
                    label_ids.append(label_map['O'])

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids.append(pad_token_label_id)

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length


            feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, tokens=tokens)

            features.append(feature)

    return features


def get_labels():
    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC","B-DRUG","I-DRUG"]
