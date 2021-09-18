from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import six
import tensorflow as tf


class AlbertConfig(object):
  """Configuration for `AlbertModel`.

  The default settings match the configuration of model `albert_xxlarge`.
  """

  def __init__(self,
               vocab_size,
               embedding_size=128,
               hidden_size=4096,
               num_hidden_layers=12,
               num_hidden_groups=1,
               num_attention_heads=64,
               intermediate_size=16384,
               inner_group_num=1,
               down_scale_factor=1,
               hidden_act="gelu",
               hidden_dropout_prob=0,
               attention_probs_dropout_prob=0,
               max_position_embeddings=512,
               type_vocab_size=2,
               initializer_range=0.02,
               backward_compatible=True):
    """Constructs AlbertConfig.
    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `AlbertModel`.
      embedding_size: size of voc embeddings.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_hidden_groups: Number of group for the hidden layers, parameters in
        the same group are shared.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      inner_group_num: int, number of inner repetition of attention and ffn.
      down_scale_factor: float, the scale to apply
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `AlbertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_groups = num_hidden_groups
    self.num_attention_heads = num_attention_heads
    self.inner_group_num = inner_group_num
    self.down_scale_factor = down_scale_factor
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = AlbertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"