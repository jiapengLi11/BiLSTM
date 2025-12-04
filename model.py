import torch
import torch.nn as nn
from config import *


class BiLSTMSimilarityModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, dense_units, dropout_rate, max_seq_len):
        super(BiLSTMSimilarityModel, self).__init__()
        self.max_seq_len = max_seq_len

        # 嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_INDEX
        )
        self.embedding_dropout = nn.Dropout(0.3)  # 新增：嵌入层后加Dropout

        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,  # 仅在num_layers>1时生效（当前num_layers=2）
            num_layers=2  # 新增：双层LSTM增强特征提取
        )

        # Dropout层
        self.dropout = nn.Dropout(0.4)

        # 全连接层（增加批归一化）修正fc1输入维度：lstm_units*2（双向）*3（hidden+avg+max）*2（两个句子）= 768
        self.fc1 = nn.Linear(lstm_units * 2 * 3 * 2, dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)  # 新增：批归一化
        self.fc2 = nn.Linear(dense_units, dense_units // 2)  # 新增：中间层
        self.bn2 = nn.BatchNorm1d(dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

        # 激活函数（用LeakyReLU缓解梯度消失）
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, sent1, sent2):
        # 嵌入层 + dropout
        embed1 = self.embedding(sent1)
        embed1 = self.embedding_dropout(embed1)
        embed2 = self.embedding(sent2)
        embed2 = self.embedding_dropout(embed2)

        # BiLSTM层（获取序列输出而非仅最后hidden）
        out1, (hidden1, _) = self.bilstm(embed1)
        out2, (hidden2, _) = self.bilstm(embed2)

        # 结合最后hidden和序列均值（增强特征）
        # 统一为第一个句子的方式：拼接hidden、avg和max三种特征
        hidden1 = torch.cat((hidden1[0], hidden1[1]), dim=1)
        avg1 = torch.mean(out1, dim=1)
        max1 = torch.max(out1, dim=1)[0]  # 序列最大值
        feat1 = torch.cat((hidden1, avg1, max1), dim=1)  # 拼接3种特征

        # 第二个句子也采用相同的方式
        hidden2 = torch.cat((hidden2[0], hidden2[1]), dim=1)
        avg2 = torch.mean(out2, dim=1)
        max2 = torch.max(out2, dim=1)[0]  # 序列最大值
        feat2 = torch.cat((hidden2, avg2, max2), dim=1)  # 拼接3种特征

        # Dropout
        feat1 = self.dropout(feat1)
        feat2 = self.dropout(feat2)

        # 拼接两个句子的特征
        combined = torch.cat((feat1, feat2), dim=1)

        # 全连接层 + 批归一化 + dropout
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        output = self.fc3(x)
        return output


def build_bilstm_similarity_model(vocab_size, max_seq_len):
    """构建中文语义相似度BiLSTM双塔模型（回归任务）"""
    model = BiLSTMSimilarityModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        max_seq_len=max_seq_len
    ).to(DEVICE)

    return model


if __name__ == "__main__":
    model = build_bilstm_similarity_model(vocab_size=VOCAB_SIZE, max_seq_len=10)
    print("模型结构：")
    print(model)

    # 测试模型输入输出
    sent1 = torch.randint(0, 1000, (32, 10)).to(DEVICE)
    sent2 = torch.randint(0, 1000, (32, 10)).to(DEVICE)
    output = model(sent1, sent2)
    print(f"输入形状: {sent1.shape}, {sent2.shape}")
    print(f"输出形状: {output.shape}")
