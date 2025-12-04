import re
import jieba
import torch
import numpy as np
import os
from config import *
from model import build_bilstm_similarity_model
from preprocessor import Tokenizer, clean_chinese_text, tokenize_chinese, pad_sequences


def load_model_and_tokenizer():
    """加载保存的模型和Tokenizer（带完整异常处理）"""
    # ========== 校验文件是否存在 ==========
    if not os.path.exists(TOKENIZER_SAVE_PATH):
        raise FileNotFoundError(f"Tokenizer文件不存在：{TOKENIZER_SAVE_PATH}\n请先运行train.py训练并保存Tokenizer！")

    if not os.path.exists(MODEL_SAVE_PATH) and not os.path.exists(MODEL_SAVE_PATH.replace('.pth', '_best.pth')):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_SAVE_PATH}\n请先运行train.py训练模型！")

    # ========== 加载Tokenizer ==========
    try:
        tokenizer = Tokenizer.load(TOKENIZER_SAVE_PATH)
        print(f"✅ 已加载Tokenizer：{TOKENIZER_SAVE_PATH}")
    except Exception as e:
        raise RuntimeError(f"Tokenizer加载失败：{e}\n可能是文件损坏或版本不兼容，请重新训练生成！")

    # ========== 加载模型 ==========
    model = build_bilstm_similarity_model(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    # 优先加载最佳模型
    best_model_path = MODEL_SAVE_PATH.replace('.pth', '_best.pth')
    load_success = False

    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ 已加载最佳模型：{best_model_path}")
            load_success = True
        except Exception as e:
            print(f"⚠️  最佳模型加载失败：{e}，尝试加载最终模型...")

    if not load_success and os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            print(f"✅ 已加载最终模型：{MODEL_SAVE_PATH}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n模型结构与参数不匹配，请重新训练模型！")

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# 全局变量初始化
model = None
tokenizer = None


def predict_similarity(sent1, sent2):
    """预测两个中文句子的相似度（0-5分 + 归一化0-1分）"""
    global model, tokenizer

    # 首次调用时加载模型和Tokenizer
    if model is None or tokenizer is None:
        print("正在加载模型和Tokenizer...")
        model, tokenizer = load_model_and_tokenizer()

    # ========== 预处理新句子 ==========
    print("\n" + "=" * 60)
    print("预处理新句子...")

    # 清洗
    sent1_clean = clean_chinese_text(sent1)
    sent2_clean = clean_chinese_text(sent2)
    print(f"清洗后：\n句子1：{sent1_clean}\n句子2：{sent2_clean}")

    # 分词
    sent1_token = tokenize_chinese(sent1_clean)
    sent2_token = tokenize_chinese(sent2_clean)
    print(f"分词后：\n句子1：{sent1_token}\n句子2：{sent2_token}")

    # 编码
    try:
        sent1_enc = tokenizer.texts_to_sequences([sent1_token])
        sent2_enc = tokenizer.texts_to_sequences([sent2_token])
    except Exception as e:
        raise RuntimeError(f"文本编码失败：{e}\n请检查Tokenizer是否正常加载！")

    # 填充
    sent1_pad = pad_sequences(sent1_enc, maxlen=MAX_SEQ_LEN)
    sent2_pad = pad_sequences(sent2_enc, maxlen=MAX_SEQ_LEN)
    print(f"填充后形状：\n句子1：{sent1_pad.shape}\n句子2：{sent2_pad.shape}")

    # ========== 转换为张量并预测 ==========
    sent1_tensor = torch.tensor(sent1_pad, dtype=torch.long).to(DEVICE)
    sent2_tensor = torch.tensor(sent2_pad, dtype=torch.long).to(DEVICE)

    print("开始预测...")
    with torch.no_grad():
        similarity_score = model(sent1_tensor, sent2_tensor).item()

    # ========== 结果处理 ==========
    similarity_score = max(0.0, min(5.0, similarity_score))  # 限制范围
    normalized_score = similarity_score / 5.0  # 归一化

    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("相似度预测结果")
    print("=" * 60)
    print(f"句子1：{sent1}")
    print(f"句子2：{sent2}")
    print(f"原始相似度分数（0-5分）：{similarity_score:.2f}")
    print(f"归一化相似度（0-1分）：{normalized_score:.4f}")

    # 相似度判定
    if normalized_score >= 0.8:
        judgment = "高度相似"
    elif normalized_score >= 0.6:
        judgment = "较相似"
    elif normalized_score >= 0.4:
        judgment = "中等相似"
    elif normalized_score >= 0.2:
        judgment = "低相似"
    else:
        judgment = "不相似"
    print(f"语义相似度判定：{judgment}")
    print("=" * 60)


if __name__ == "__main__":
    # 输入句子
    sentence1 = "一个女人在测量身高"
    sentence2 = "一位女士在量身高"

    # 预测
    try:
        predict_similarity(sentence1, sentence2)

        # 批量预测示例（可选）
        # print("\n" + "-"*60 + "\n批量预测：")
        # test_pairs = [
        #     ("一个人在跳广场舞", "一群人在跳广场舞"),
        #     ("猫喜欢吃鱼", "狗喜欢啃骨头"),
        #     ("今天天气很好", "今日阳光明媚"),
        # ]
        # for s1, s2 in test_pairs:
        #     predict_similarity(s1, s2)
        #     print("\n" + "-"*60 + "\n")
    except Exception as e:
        print(f"\n❌ 预测失败：{e}")