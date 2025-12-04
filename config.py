# ================================= 路径配置 =================================
TRAIN_DATA_PATH = r"C:\Users\22842\Desktop\text_similarity\STS-B\STS-B.train.data"
VALID_DATA_PATH = r"C:\Users\22842\Desktop\text_similarity\STS-B\STS-B.valid.data"
TEST_DATA_PATH = r"C:\Users\22842\Desktop\text_similarity\STS-B\STS-B.test.data"

MODEL_SAVE_PATH = "./chinese_similarity_model.pth"
TOKENIZER_SAVE_PATH = "./chinese_tokenizer.pkl"
TRAIN_HISTORY_SAVE_PATH = "./training_history.png"

# ================================= 超参数配置 =================================
VOCAB_SIZE = 5000  # 词汇表大小
MAX_SEQ_LEN = 20  # 序列最大长度（自动计算95分位数）
OOV_TOKEN = "<UNK>"  # 未登录词标记
PAD_TOKEN = "<PAD>"  # 填充词标记
PAD_INDEX = 0  # PAD对应的索引
OOV_INDEX = 1  # OOV对应的索引

# 模型参数
EMBEDDING_DIM = 128  # 嵌入维度
LSTM_UNITS = 64  # BiLSTM隐藏层单元数
DROPOUT_RATE = 0.2  # Dropout比例
DENSE_UNITS = 64  # 全连接层单元数

# 训练参数
EPOCHS = 50  # 训练轮数
BATCH_SIZE = 64  # 批次大小
LEARNING_RATE = 0.0005  # 学习率
WEIGHT_DECAY = 1e-4  # L2正则化（防过拟合）

# 数据集划分参数
TEST_SIZE = 0.2  # 测试集比例
VALID_SIZE = 0.5  # 验证集比例（从测试集中拆分）
RANDOM_STATE = 42  # 固定随机种子
# 早停配置
PATIENCE = 8  # 多少个epoch验证集loss不下降就停止
MIN_DELTA = 1e-5  # 最小改善幅度（小于这个值不算改善）

# ================================= PyTorch配置 =================================
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测GPU/CPU
NUM_WORKERS = 4  # DataLoader多进程数（CPU核心数）
PIN_MEMORY = True  # 内存固定（GPU加速）
