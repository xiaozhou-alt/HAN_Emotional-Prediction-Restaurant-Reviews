import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionMetrics:
    @staticmethod
    def calculate(y_true, y_pred):
        """
        输入形状均为(batch_size, 4)的tensor
        返回四个维度的平均指标和每个维度的单独指标
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        metrics = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score
        }
        
        results = {}
        # 计算每个维度的指标
        for dim, name in enumerate(['overall', 'env', 'flavor', 'service']):
            for metric_name, metric_fn in metrics.items():
                key = f"{name}_{metric_name}"
                results[key] = metric_fn(y_true[:, dim], y_pred[:, dim])
        
        # 计算全局平均指标
        for metric_name in metrics.keys():
            results[f"avg_{metric_name}"] = np.mean(
                [results[f"{name}_{metric_name}"] 
                for name in ['overall', 'env', 'flavor', 'service']]
            )
        
        return results