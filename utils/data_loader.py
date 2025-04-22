import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

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