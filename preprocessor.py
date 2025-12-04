import re
import jieba
import numpy as np
import pickle
from collections import defaultdict
from config import *


class Tokenizer:
    def __init__(self, num_words=VOCAB_SIZE, oov_token=OOV_TOKEN):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: OOV_INDEX, PAD_TOKEN: PAD_INDEX}
        self.index_word = {OOV_INDEX: oov_token, PAD_INDEX: PAD_TOKEN}
        self.word_counts = defaultdict(int)
        self.vocab_size = 2  # 初始包含PAD和OOV
        self.max_seq_len = None  # 新增：初始化max_seq_len属性

    def fit_on_texts(self, texts):
        """基于文本列表构建词汇表"""
        for tokens in texts:
            for token in tokens:
                self.word_counts[token] += 1

        # 按词频排序并添加到词汇表
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_words:
            if self.vocab_size < self.num_words:
                self.word_index[word] = self.vocab_size
                self.index_word[self.vocab_size] = word
                self.vocab_size += 1
            else:
                break

    def texts_to_sequences(self, texts):
        """将文本转换为索引序列"""
        sequences = []
        for tokens in texts:
            sequence = []
            for token in tokens:
                sequence.append(self.word_index.get(token, OOV_INDEX))
            sequences.append(sequence)
        return sequences

    def save(self, save_path):
        """将Tokenizer对象序列化保存到文件"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'word_index': self.word_index,
                'index_word': self.index_word,
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len  # （可选）保存训练时的序列长度
            }, f)
        print(f"Tokenizer已保存至：{save_path}")

    @classmethod
    def load(cls, load_path):
        """从文件加载Tokenizer对象"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        # 重建Tokenizer实例
        tokenizer = cls()
        tokenizer.word_index = data['word_index']
        tokenizer.index_word = data['index_word']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.max_seq_len = data.get('max_seq_len')  # 兼容旧版本

        return tokenizer


def clean_chinese_text(text):
    """中文文本清洗：去除标点、特殊字符、多余空格"""
    text = str(text).strip()
    text = re.sub(r"[^\u4e00-\u9fa5\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_chinese(text):
    """中文分词（jieba精确模式）"""
    return jieba.lcut(text)


def pad_sequences(sequences, maxlen, padding='post', truncating='post'):
    """序列填充/截断（增加maxlen校验）"""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)  # 兜底方案：取最长序列长度

    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_length = maxlen - len(seq)
            if padding == 'post':
                seq = seq + [PAD_INDEX] * pad_length
            else:
                seq = [PAD_INDEX] * pad_length + seq
        padded_sequences.append(seq)
    return np.array(padded_sequences, dtype=np.int64)


def preprocess_all_data(train_data, valid_data, test_data):
    """预处理所有数据集（训练集+验证集+测试集）"""
    # 解包数据
    (train_sent1, train_sent2, train_scores) = train_data
    (valid_sent1, valid_sent2, valid_scores) = valid_data
    (test_sent1, test_sent2, test_scores) = test_data

    # 1. 文本清洗
    print("开始文本清洗...")
    train_sent1_clean = [clean_chinese_text(s) for s in train_sent1]
    train_sent2_clean = [clean_chinese_text(s) for s in train_sent2]
    valid_sent1_clean = [clean_chinese_text(s) for s in valid_sent1]
    valid_sent2_clean = [clean_chinese_text(s) for s in valid_sent2]
    test_sent1_clean = [clean_chinese_text(s) for s in test_sent1]
    test_sent2_clean = [clean_chinese_text(s) for s in test_sent2]

    # 2. 中文分词
    print("开始中文分词...")
    train_sent1_token = [tokenize_chinese(s) for s in train_sent1_clean]
    train_sent2_token = [tokenize_chinese(s) for s in train_sent2_clean]
    valid_sent1_token = [tokenize_chinese(s) for s in valid_sent1_clean]
    valid_sent2_token = [tokenize_chinese(s) for s in valid_sent2_clean]
    test_sent1_token = [tokenize_chinese(s) for s in test_sent1_clean]
    test_sent2_token = [tokenize_chinese(s) for s in test_sent2_clean]

    # 3. 构建词汇表
    print("构建词汇表...")
    all_train_tokens = train_sent1_token + train_sent2_token
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(all_train_tokens)
    print(f"词汇表大小：{tokenizer.vocab_size}")

    # 4. 编码（词→索引序列）
    print("文本编码...")
    train_sent1_enc = tokenizer.texts_to_sequences(train_sent1_token)
    train_sent2_enc = tokenizer.texts_to_sequences(train_sent2_token)
    valid_sent1_enc = tokenizer.texts_to_sequences(valid_sent1_token)
    valid_sent2_enc = tokenizer.texts_to_sequences(valid_sent2_token)
    test_sent1_enc = tokenizer.texts_to_sequences(test_sent1_token)
    test_sent2_enc = tokenizer.texts_to_sequences(test_sent2_token)

    # 5. 序列填充/截断（统一长度）
    print("序列填充...")
    global MAX_SEQ_LEN
    all_train_seqs = train_sent1_enc + train_sent2_enc
    seq_lengths = [len(seq) for seq in all_train_seqs]
    MAX_SEQ_LEN = int(np.percentile(seq_lengths, 95))
    print(f"序列统一长度：{MAX_SEQ_LEN}（覆盖95%训练集句子）")
    # 添加这一行，将 MAX_SEQ_LEN 赋值给 tokenizer 对象
    tokenizer.max_seq_len = MAX_SEQ_LEN

    # 填充所有序列
    train_sent1_pad = pad_sequences(train_sent1_enc, maxlen=MAX_SEQ_LEN)
    train_sent2_pad = pad_sequences(train_sent2_enc, maxlen=MAX_SEQ_LEN)
    valid_sent1_pad = pad_sequences(valid_sent1_enc, maxlen=MAX_SEQ_LEN)
    valid_sent2_pad = pad_sequences(valid_sent2_enc, maxlen=MAX_SEQ_LEN)
    test_sent1_pad = pad_sequences(test_sent1_enc, maxlen=MAX_SEQ_LEN)
    test_sent2_pad = pad_sequences(test_sent2_enc, maxlen=MAX_SEQ_LEN)

    # 输出格式检查
    print(f"\n预处理完成，数据形状：")
    print(f"训练集输入：{train_sent1_pad.shape} | {train_sent2_pad.shape} | 标签：{train_scores.shape}")
    print(f"验证集输入：{valid_sent1_pad.shape} | {valid_sent2_pad.shape} | 标签：{valid_scores.shape}")
    print(f"测试集输入：{test_sent1_pad.shape} | {test_sent2_pad.shape} | 标签：{test_scores.shape}")

    return (
        (train_sent1_pad, train_sent2_pad, train_scores),
        (valid_sent1_pad, valid_sent2_pad, valid_scores),
        (test_sent1_pad, test_sent2_pad, test_scores),
        tokenizer
    )


if __name__ == "__main__":
    from data_loader import load_and_split_data

    train_data, valid_data, test_data = load_and_split_data()
    preprocessed_data = preprocess_all_data(train_data, valid_data, test_data)
