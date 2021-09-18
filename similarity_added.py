# albert获取句向量，通过最后一层sequence编码输出进行avg/max得到
# 尝试bert-whitening

import bert
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

max_seq_length = 100
albert_dir = "./albert_tiny"
vocab_file = os.path.join(albert_dir, "vocab.txt")
tokenizer = bert.albert_tokenization.FullTokenizer(vocab_file=vocab_file)
model_params = bert.params_from_pretrained_ckpt(albert_dir)
albert_model = bert.BertModelLayer.from_params(model_params, name="albert")
albert_model(tf.zeros((1, 128)))
bert.load_albert_weights(albert_model, albert_dir)


def cos_similar(sen_a_vec, sen_b_vec):
    """
    计算两个句子的余弦相似度
    """
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


def load_albert_zh_model(method: str):
    tokens1 = tokenizer.tokenize("你好世界")
    token_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
    print(token_ids1)
    tokens2 = tokenizer.tokenize("你好中国")
    token_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
    print(token_ids2)
    tokens3 = tokenizer.tokenize("早上好呀")
    token_ids3 = tokenizer.convert_tokens_to_ids(tokens3)
    print(token_ids3)
    tokens4 = tokenizer.tokenize("带好食物")
    token_ids4 = tokenizer.convert_tokens_to_ids(tokens4)
    print(token_ids4)
    seq_vector1 = albert_model.call(tf.constant([token_ids1, token_ids4]))
    seq_vector2 = albert_model.call(tf.constant([token_ids2, token_ids3]))
    seq_vector3 = albert_model.call(tf.constant([token_ids2]))
    if method == "max":
        sent_vector1 = tf.math.reduce_max(seq_vector1, axis=1)
        sent_vector2 = tf.math.reduce_max(seq_vector2, axis=1)
        sent_vector3 = tf.math.reduce_max(seq_vector3, axis=1)
    elif method == "avg":
        sent_vector1 = tf.math.reduce_mean(seq_vector1, axis=1)
        sent_vector2 = tf.math.reduce_mean(seq_vector2, axis=1)
        sent_vector3 = tf.math.reduce_mean(seq_vector3, axis=1)
    else:
        raise ValueError("compute vector method must be max or avg")
    print(sent_vector2.shape)
    # print(cos_similar(sent_vector1, sent_vector2))
    print(cosine_similarity(sent_vector1, sent_vector2))
    print(cosine_similarity(sent_vector1, sent_vector3))


def get_input_tokens_id(sentences: list):
    input_feature = []
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        if len(tokens) > max_seq_length:
            tokens = tokens[0:max_seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids.extend([0] * (max_seq_length - len(token_ids)))
        input_feature.append(token_ids)
    return tf.constant(input_feature)


def get_sent_vector(seq_vec, sent_lens, method="max"):
    sent_vec_list = []
    for idx in range(len(sent_lens)):
        seq_vec_no_padding = seq_vec[idx][:sent_lens[idx]]
        if method == "max":
            sent_vector = tf.math.reduce_max(seq_vec_no_padding, axis=0)
        elif method == "avg":
            sent_vector = tf.math.reduce_mean(seq_vec_no_padding, axis=0)
        else:
            raise ValueError("compute vector method must be max or avg")
        sent_vec_list.append(sent_vector)
    return sent_vec_list


def get_vect(seq_vec, method="max"):
    if method == "max":
        sent_vector = tf.math.reduce_max(seq_vec, axis=1)

    elif method == "avg":
        sent_vector = tf.math.reduce_mean(seq_vec, axis=1)

    else:
        raise ValueError("compute vector method must be max or avg")
    return sent_vector


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def similar_score(source_content, target_content):
    """
    :param  source_content: 模版话术
    :param target_content: 对话记录
    :return 相似度分数
    """
    if isinstance(source_content, str):
        source_content = source_content.strip().split("。")[:-1]
    if isinstance(target_content, str):
        target_content = target_content.strip().split("。")[:-1]

    source_sent_lens = [len(sent) for sent in source_content]
    target_sent_lens = [len(sent) for sent in target_content]

    source_input_feat = get_input_tokens_id(source_content)
    source_seq_output = albert_model.call(source_input_feat)
    source_sent_vectors = get_sent_vector(source_seq_output, source_sent_lens)        # B*hidden_size

    target_input_feat = get_input_tokens_id(target_content)
    target_seq_output = albert_model.call(target_input_feat)
    target_sent_vectors = get_sent_vector(target_seq_output, target_sent_lens)        # B*hidden_size

    # 源句子数*目标句子数 第一行表示目标句子中的每一个句子与源句子中第一条的相似度
    score_matrix = cosine_similarity(source_sent_vectors, target_sent_vectors)
    print(score_matrix)

    # bert-whitening
    all_vectors = [tf.reshape(source_sent_vectors[idx], [1, -1]) for idx in range(len(source_sent_vectors))]
    for idx in range(len(target_sent_vectors)):
        all_vectors.append(tf.reshape(target_sent_vectors[idx], [1, -1]))
    kernel, bias = compute_kernel_bias(all_vectors)

    source_sent_whitening = []
    for idx in range(len(source_sent_vectors)):
        source_sent_whitening.append(transform_and_normalize(source_sent_vectors[idx].numpy(), kernel, bias).flatten())

    target_sent_whitening = []
    for idx in range(len(target_sent_vectors)):
        target_sent_whitening.append(transform_and_normalize(target_sent_vectors[idx].numpy(), kernel, bias).flatten())
    sims = (source_sent_vectors[0].numpy().reshape(1,-1) * target_sent_vectors[0].numpy().reshape(1,-1)).sum(axis=1)
    print(sims)
    # score_matrix = cosine_similarity(source_sent_whitening, target_sent_whitening)
    # print(score_matrix)


if __name__ == "__main__":
    # load_albert_zh_model("avg")
    source = "您好，请问有什么能帮到您的。我想换一个套餐。好的。您还有其他问题吗。没有了。好的，再见，祝您生活愉快。"
    target = "你好，能听见吗，有啥事。我想换一个套餐。行，流量和通话分钟数有要求吗。没有。好的，就这样吧，再见。"
    # target = "这个函数的输入是n个长度相同的list或者array。函数的处理是计算这n个list两两之间的余弦相似性。最后生成的相似矩阵中的表示的是原来输入的矩阵中的第i行和第j行两个向量的相似性。所以生成的是n*n的相似性矩阵。"
    similar_score(source, target)
