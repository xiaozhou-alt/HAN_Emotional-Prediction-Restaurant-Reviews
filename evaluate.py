import torch
from utils.data_loader import get_dataloader
from utils.metrics import RegressionMetrics
from tqdm import tqdm
import yaml
import pickle
from models.han import HierarchicalAttentionNetwork

def evaluate(model_path, config_path="configs/base.yaml"):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    with open("data/processed/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    model = HierarchicalAttentionNetwork(
        vocab_size=len(vocab),
        embed_dim=config['model']['embed_dim'],
        word_hidden_dim=config['model']['word_hidden_dim'],
        sentence_hidden_dim=config['model']['sentence_hidden_dim'],
        dropout=config['model']['dropout']
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载测试数据
    test_loader = get_dataloader(
        "data/processed/test.pkl", 
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 评估过程
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            all_preds.append(outputs.cpu())
            all_labels.append(y_batch.cpu())
    
    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    metrics = RegressionMetrics.calculate(final_labels, final_preds)
    
    # 打印结果
    print("\nEvaluation Results:")
    for name in ['overall', 'env', 'flavor', 'service']:
        print(f"[{name.upper()}]")
        print(f"  MSE: {metrics[f'{name}_mse']:.4f}")
        print(f"  MAE: {metrics[f'{name}_mae']:.4f}")
        print(f"  R²:  {metrics[f'{name}_r2']:.4f}")
    
    print("\nGlobal Averages:")
    print(f"MSE: {metrics['avg_mse']:.4f}")
    print(f"MAE: {metrics['avg_mae']:.4f}")
    print(f"R²:  {metrics['avg_r2']:.4f}")

if __name__ == "__main__":
    evaluate("checkpoints/best_model.pth")