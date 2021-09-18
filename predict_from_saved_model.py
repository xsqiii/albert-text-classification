import tensorflow as tf
import tokenization
from classifier_data_lib import InputFeatures
import numpy as np
import time

saved_model_path = "./saved_model/1"
max_seq_len = 128
tokenizer = tokenization.FullTokenizer(vocab_file="./albert_tiny/vocab.txt", do_lower_case=False)
label_dict = {0: "negative", 1: "positive"}


def convert_to_feature(sentences: list):
    features = []
    for sentence in sentences:
        tokens = str(sentence)

        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        label_ids = [0]
        ntokens.append("[CLS]")
        segment_ids.append(0)

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)

        ntokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_ids,
        )
        features.append(feature)

    return features


def infer(sentences: list):
    loaded = tf.saved_model.load(saved_model_path)
    predict_fn = loaded.signatures["serving_default"]
    features = convert_to_feature(sentences)
    input_feat = {"input_word_ids": tf.constant([feature.input_ids for feature in features]),
                  "input_mask": tf.constant([feature.input_mask for feature in features]),
                  "input_type_ids": tf.constant([feature.segment_ids for feature in features]), }
    _ = predict_fn(**input_feat)    # lazy initialization
    start = time.time()
    output = predict_fn(**input_feat)
    print("cost time:", time.time() - start)
    return [label_dict[val] for val in np.argmax(output["pred"], axis=1)]


if __name__ == "__main__":
    sent_list = ["水果新鲜！发货快，服务好，京东物流顶呱呱，快递小哥服务好周到，送货上门，每次都热情满满，辛苦了，必须赞！",
                 "隔音差，加速有点不给力",
                 "最满意的是耗电量很低，百公里耗电才11.8度，这样算起来，出行成本比地铁的还低。"
                 ]
    print(infer(sent_list))
