import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import *


class TextSimilarityDataset(Dataset):
    def __init__(self, sent1, sent2, scores):
        self.sent1 = sent1
        self.sent2 = sent2
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            'sent1': self.sent1[idx],
            'sent2': self.sent2[idx],
            'score': self.scores[idx]
        }


def load_single_data(file_path):
    """加载单个 .data 文件（制表符分隔）"""
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            encoding='utf-8',
            usecols=[0, 1, 2]
        )
        df.columns = ["sent1", "sent2", "score"]
        df = df.dropna(subset=["sent1", "sent2", "score"])
        df["score"] = df["score"].astype(float)
        df = df[(df["score"] >= 0) & (df["score"] <= 5)]

        print(f"加载 {file_path}：{len(df)} 条有效数据，分数范围：{df['score'].min():.1f}~{df['score'].max():.1f}")
        return df["sent1"].tolist(), df["sent2"].tolist(), df["score"].values

    except Exception as e:
        print(f"加载数据失败：{e}")
        print("请检查：1. 文件路径是否正确 2. 文件是否为制表符分隔 3. 前3列是否为sent1、sent2、score")
        raise


def load_and_split_data():
    """加载所有数据，并按 8:1:1 划分训练集/验证集/测试集"""
    train_sent1, train_sent2, train_scores = load_single_data(TRAIN_DATA_PATH)
    valid_sent1, valid_sent2, valid_scores = load_single_data(VALID_DATA_PATH)
    test_sent1, test_sent2, test_scores = load_single_data(TEST_DATA_PATH)

    # （可选）若只有训练集，用下面代码拆分（注释上面3行，取消下面注释）
    # all_sent1, all_sent2, all_scores = load_single_data(TRAIN_DATA_PATH)
    # train_sent1, temp_sent1, train_sent2, temp_sent2, train_scores, temp_scores = train_test_split(
    #     all_sent1, all_sent2, all_scores, test_size=TEST_SIZE, random_state=RANDOM_STATE
    # )
    # valid_sent1, test_sent1, valid_sent2, test_sent2, valid_scores, test_scores = train_test_split(
    #     temp_sent1, temp_sent2, temp_scores, test_size=VALID_SIZE, random_state=RANDOM_STATE
    # )

    print(f"\n数据集划分：")
    print(f"训练集：{len(train_sent1)} 条")
    print(f"验证集：{len(valid_sent1)} 条")
    print(f"测试集：{len(test_sent1)} 条")

    return (
        (train_sent1, train_sent2, train_scores),
        (valid_sent1, valid_sent2, valid_scores),
        (test_sent1, test_sent2, test_scores)
    )


def create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=BATCH_SIZE):
    """创建DataLoader"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_data, valid_data, test_data = load_and_split_data()
    train_sent1, train_sent2, train_scores = train_data
    print(f"\n前2条训练集样本：")
    print(f"句1：{train_sent1[0]} | 句2：{train_sent2[0]} | 分数：{train_scores[0]:.2f}")
    print(f"句1：{train_sent1[1]} | 句2：{train_sent2[1]} | 分数：{train_scores[1]:.2f}")