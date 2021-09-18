# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""BERT library to process data for classification task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os

from absl import logging, flags
import tensorflow as tf

import tokenization

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @staticmethod
    def get_processor_name():
        """Gets the string identifier of the processor."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SentimentProcessor(DataProcessor):
    """Processor for the sentiment Classificaition data set"""

    def __init__(self):
        self.language = "zh"

    def read_file(self, input_file):
        with open(input_file, "r") as f:
            lines = []
            for line in f.readlines():
                label, sentence = line.strip().split("\t")
                lines.append([sentence, label])
            return lines

    def get_labels(self):
        return ["0", "1"]

    def get_train_examples(self, data_dir):
        return self._create_example(self.read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.read_file(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(self.read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_processor_name(self):
        return "sentiment"

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    if FLAGS.classification_task_name.lower() != "sts":
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in ALBERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if FLAGS.classification_task_name.lower() != "sts":
        label_id = label_map[example.label]
    else:
        label_id = example.label

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d", ex_index, len(examples))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_float_feature([feature.label_id]) \
            if FLAGS.classification_task_name.lower() == "sts" else create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def generate_tf_record_from_data_file(processor,
                                      data_dir,
                                      spm_model_file,
                                      train_data_output_path=None,
                                      eval_data_output_path=None,
                                      max_seq_length=128,
                                      do_lower_case=True):
    """Generates and saves training data into a tf record file.

    Arguments:
        processor: Input processor object to be used for generating data. Subclass
          of `DataProcessor`.
        data_dir: Directory that contains train/eval data to process. Data files
          should be in from "dev.tsv", "test.tsv", or "train.tsv".
        vocab_file: Text file with words to be used for training/evaluation.
        train_data_output_path: Output to which processed tf record for training
          will be saved.
        eval_data_output_path: Output to which processed tf record for evaluation
          will be saved.
        max_seq_length: Maximum sequence length of the to be generated
          training/eval data.
        do_lower_case: Whether to lower case input text.

    Returns:
        A dictionary containing input meta data.
    """
    assert train_data_output_path or eval_data_output_path

    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=spm_model_file, do_lower_case=do_lower_case)
    assert train_data_output_path
    train_input_data_examples = processor.get_train_examples(data_dir)
    file_based_convert_examples_to_features(train_input_data_examples, label_list,
                                            max_seq_length, tokenizer,
                                            train_data_output_path)
    num_training_data = len(train_input_data_examples)

    if eval_data_output_path:
        eval_input_data_examples = processor.get_dev_examples(data_dir)
        file_based_convert_examples_to_features(eval_input_data_examples,
                                                label_list, max_seq_length,
                                                tokenizer, eval_data_output_path)

    meta_data = {
        "task_type": "albert_classification",
        "processor_type": processor.get_processor_name(),
        "num_labels": len(processor.get_labels()),
        "train_data_size": num_training_data,
        "max_seq_length": max_seq_length,
    }

    if eval_data_output_path:
        meta_data["eval_data_size"] = len(eval_input_data_examples)

    return meta_data


def generate_test_tf_record_from_data_file(processor,
                                           data_dir,
                                           spm_model_file,
                                           test_data_output_path=None,
                                           max_seq_length=128,
                                           do_lower_case=True):
    assert test_data_output_path
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=spm_model_file, do_lower_case=do_lower_case)
    test_input_data_examples = processor.get_test_examples(data_dir)
    file_based_convert_examples_to_features(test_input_data_examples, label_list,
                                            max_seq_length, tokenizer,
                                            test_data_output_path)