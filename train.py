import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import time  # 新增：用于计时
from config import *
from data_loader import load_and_split_data, TextSimilarityDataset, create_data_loaders
from preprocessor import preprocess_all_data
from model import build_bilstm_similarity_model


# ---------------------- 新增：训练曲线可视化函数 ----------------------
def plot_training_history(history):
    """
    绘制并保存训练曲线：MSE损失曲线 + MAE曲线
    history: 包含train_loss, val_loss, train_mae, val_mae的字典
    """
    # 设置中文字体（避免中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 创建2个子图（MSE + MAE）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------- 子图1：MSE损失曲线 ----------------------
    ax1.plot(history['train_loss'], label='训练集MSE', color='#FF6B6B', linewidth=2)
    ax1.plot(history['val_loss'], label='验证集MSE', color='#4ECDC4', linewidth=2)
    ax1.set_title('训练/验证集MSE损失变化', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE损失', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)  # 损失从0开始

    # 标记最佳验证集MSE的位置
    best_val_loss_idx = np.argmin(history['val_loss'])
    ax1.scatter(best_val_loss_idx, history['val_loss'][best_val_loss_idx],
                color='red', s=60, zorder=5, label='最佳验证MSE')
    ax1.annotate(f'最低MSE: {history["val_loss"][best_val_loss_idx]:.3f}',
                 xy=(best_val_loss_idx, history['val_loss'][best_val_loss_idx]),
                 xytext=(best_val_loss_idx + 2, history['val_loss'][best_val_loss_idx] + 0.2),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # ---------------------- 子图2：MAE曲线 ----------------------
    ax2.plot(history['train_mae'], label='训练集MAE', color='#FF6B6B', linewidth=2)
    ax2.plot(history['val_mae'], label='验证集MAE', color='#4ECDC4', linewidth=2)
    ax2.set_title('训练/验证集MAE变化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 标记最佳验证集MAE的位置
    best_val_mae_idx = np.argmin(history['val_mae'])
    ax2.scatter(best_val_mae_idx, history['val_mae'][best_val_mae_idx],
                color='red', s=60, zorder=5, label='最佳验证MAE')
    ax2.annotate(f'最低MAE: {history["val_mae"][best_val_mae_idx]:.3f}',
                 xy=(best_val_mae_idx, history['val_mae'][best_val_mae_idx]),
                 xytext=(best_val_mae_idx + 2, history['val_mae'][best_val_mae_idx] + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # 整体布局调整
    plt.tight_layout()

    # 保存图片（高分辨率）
    plt.savefig(TRAIN_HISTORY_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 训练曲线已保存至：{TRAIN_HISTORY_SAVE_PATH}")


def train_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in train_loader:  # 移除tqdm，统一在外部显示进度
        sent1 = batch['sent1'].to(DEVICE)
        sent2 = batch['sent2'].to(DEVICE)
        scores = batch['score'].to(DEVICE).float()

        optimizer.zero_grad()

        outputs = model(sent1, sent2).squeeze()
        loss = criterion(outputs, scores)

        loss.backward()
        optimizer.step()

        # 计算MAE
        mae = torch.mean(torch.abs(outputs - scores))

        total_loss += loss.item() * sent1.size(0)
        total_mae += mae.item() * sent1.size(0)
        total_samples += sent1.size(0)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae


def evaluate(model, data_loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            sent1 = batch['sent1'].to(DEVICE)
            sent2 = batch['sent2'].to(DEVICE)
            scores = batch['score'].to(DEVICE).float()

            outputs = model(sent1, sent2).squeeze()
            loss = criterion(outputs, scores)

            # 计算MAE
            mae = torch.mean(torch.abs(outputs - scores))

            total_loss += loss.item() * sent1.size(0)
            total_mae += mae.item() * sent1.size(0)
            total_samples += sent1.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(scores.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae, np.array(all_preds), np.array(all_labels)


def main():
    # 1. 加载并划分数据集
    print("=" * 60)
    print("第一步：加载数据集")
    print("=" * 60)
    train_data, valid_data, test_data = load_and_split_data()

    # 2. 文本预处理
    print("\n" + "=" * 60)
    print("第二步：文本预处理")
    print("=" * 60)
    preprocessed_train, preprocessed_valid, preprocessed_test, tokenizer = preprocess_all_data(
        train_data, valid_data, test_data
    )
    train_sent1, train_sent2, train_scores = preprocessed_train
    valid_sent1, valid_sent2, valid_scores = preprocessed_valid
    test_sent1, test_sent2, test_scores = preprocessed_test
    # 保存 Tokenizer（通常在 preprocess_all_data 函数末尾或 train.py 中显式保存）
    tokenizer.save(TOKENIZER_SAVE_PATH)  # TOKENIZER_SAVE_PATH 对应 ./chinese_tokenizer.pkl
    # 3. 创建数据集和数据加载器
    train_dataset = TextSimilarityDataset(train_sent1, train_sent2, train_scores)
    valid_dataset = TextSimilarityDataset(valid_sent1, valid_sent2, valid_scores)
    test_dataset = TextSimilarityDataset(test_sent1, test_sent2, test_scores)

    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset
    )

    # 4. 构建模型
    print("\n" + "=" * 60)
    print("第三步：构建模型")
    print("=" * 60)
    model = build_bilstm_similarity_model(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )
    print(model)

    # 5. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.3, min_lr=1e-6)

    # 6. 初始化早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0  # 记录验证集loss不改善的次数
    early_stop = False  # 早停标志
    history = {
        'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': []
    }

    # 7. 模型训练
    print("\n" + "=" * 60)
    print("第四步：开始训练")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(EPOCHS):
        if early_stop:
            print(f"\n早停触发！停止训练（最佳epoch：{epoch - patience_counter}）")
            break

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        # 训练
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f}")

        # 验证
        val_loss, val_mae, _, _ = evaluate(model, valid_loader, criterion)
        print(f"Valid Loss: {val_loss:.4f} | Valid MAE: {val_mae:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        # 早停逻辑
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }, MODEL_SAVE_PATH.replace('.pth', '_best.pth'))
            print(f"✅ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  验证集loss未改善，耐心剩余：{PATIENCE - patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                early_stop = True

    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\n训练总时长：{training_time // 60:.0f}分{training_time % 60:.0f}秒")

    # ---------------------- 新增：绘制训练曲线 ----------------------
    plot_training_history(history)

    # 8. 加载最佳模型
    print("\n" + "=" * 60)
    print("第五步：加载最佳模型")
    print("=" * 60)
    checkpoint = torch.load(MODEL_SAVE_PATH.replace('.pth', '_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    print(f"加载第 {best_epoch + 1} epoch 的最佳模型（Val Loss: {checkpoint['best_val_loss']:.4f}）")

    # 9. 模型评估
    print("\n" + "=" * 60)
    print("第六步：模型评估")
    print("=" * 60)
    test_mse, test_mae, test_pred, test_scores = evaluate(model, test_loader, criterion)
    pearson_corr, p_value = pearsonr(test_scores.flatten(), test_pred.flatten())

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {p_value:.4e})")

    # 10. 保存最终模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    tokenizer.save(TOKENIZER_SAVE_PATH)
    print(f"\n最终模型保存至：{MODEL_SAVE_PATH}")
    print(f"Tokenizer保存至：{TOKENIZER_SAVE_PATH}")


if __name__ == "__main__":
    main()
