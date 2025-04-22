import torch
import yaml
import pickle
import os
import pandas as pd
from models.han import HierarchicalAttentionNetwork
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # 兼容性处理（处理旧数据）
        if isinstance(data['y'], np.ndarray) and data['y'].dtype == np.object_:
            self.y = torch.FloatTensor(data['y'].astype(np.float32))
        else:
            self.y = torch.FloatTensor(data['y'])
        
        self.X = torch.LongTensor(data['X'])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(data_path, batch_size, shuffle=True):
    dataset = RatingDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def calculate_r2_score(y_true, y_pred):
    # 手动实现R²计算[2,4](@ref)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_total)


def train():
    # Load config
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    
    # Prepare data
    train_loader = get_dataloader(
        "data/processed/train.pkl", 
        config['training']['batch_size']
    )
    
    # Init model
    with open("data/processed/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = HierarchicalAttentionNetwork(
        vocab_size=len(vocab),
        embed_dim=config['model']['embed_dim'],
        word_hidden_dim=config['model']['word_hidden_dim'],
        sentence_hidden_dim=config['model']['sentence_hidden_dim'],
        dropout=config['model']['dropout']
    )
    optimizer = Adam(model.parameters(), lr=config['training']['lr'])
    criterion = torch.nn.MSELoss()
    
    # Setup directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    tb_writer = SummaryWriter("logs")
    best_loss = float('inf')
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化指标记录
    metrics_df = pd.DataFrame(columns=['Epoch', 'Loss', 'MSE', 'MAE', 'R2'])
    excel_path = "training_metrics.xlsx"
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        mse_values = []
        mae_values = []
        y_true_list = []
        y_pred_list = []
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = sum([criterion(outputs[:,i], y_batch[:,i]) for i in range(4)])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                y_pred = outputs.detach().cpu().numpy()
                y_true = y_batch.cpu().numpy()
                
                # 计算MAE[4](@ref)
                mae = np.mean(np.abs(y_true - y_pred))
                mae_values.append(mae)
                
                # 收集数据用于R2计算
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

        # 计算指标
        avg_loss = total_loss / len(train_loader)
        avg_mse = avg_loss  # 假设使用MSE作为损失函数[1](@ref)
        avg_mae = np.mean(mae_values)
        
        # 合并所有batch的数据计算R2[2,4](@ref)
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        r2 = calculate_r2_score(y_true_all, y_pred_all)

        # 记录到DataFrame
        new_row = pd.DataFrame({
            'Epoch': [epoch+1],
            'Loss': [avg_loss],
            'MSE': [avg_mse],
            'MAE': [avg_mae],
            'R2': [r2]})
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # 保存到Excel（每次epoch都保存最新结果）[6,7,8](@ref)
        if epoch == 0:
            metrics_df.to_excel(excel_path, index=False)
        else:
            with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                metrics_df.iloc[[-1]].to_excel(writer, index=False, header=False, startrow=epoch+1)
        
        tb_writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save checkpoints
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"New best model saved with loss {best_loss:.4f}")
    
    writer.close()

if __name__ == "__main__":
    train()