import torch
import jieba
import pickle
import sys  # 新增sys模块导入
from models.han import HierarchicalAttentionNetwork
import yaml
from preprocessed.preprocess import split_sentences


class RatingPredictor:
    def __init__(self, config_path="configs/base.yaml"):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Load vocab
        with open("data/processed/vocab.pkl", "rb") as f:
            self.word2idx = pickle.load(f)
        
        # Init model
        self.model = HierarchicalAttentionNetwork(
            vocab_size=len(self.word2idx),
            embed_dim=self.config['model']['embed_dim'],
            word_hidden_dim=self.config['model']['word_hidden_dim'],
            sentence_hidden_dim=self.config['model']['sentence_hidden_dim'],
            dropout=self.config['model']['dropout']
        )
        self.model.load_state_dict(torch.load("checkpoints/best_model.pth"))
        self.model.eval()
    
    def preprocess(self, text):
        sentences = split_sentences(text)[:self.config['data']['max_sentences']]
        encoded = []
        for sent in sentences:
            words = list(jieba.cut(sent))[:self.config['data']['max_words']]
            indices = [self.word2idx.get(w, 1) for w in words]
            indices += [0]*(self.config['data']['max_words'] - len(indices))
            encoded.append(indices)
        encoded += [[0]*self.config['data']['max_words']] * (self.config['data']['max_sentences'] - len(encoded))
        return torch.LongTensor([encoded[:self.config['data']['max_sentences']]])
    
    def predict(self, text):
        with torch.no_grad():
            inputs = self.preprocess(text)
            outputs = self.model(inputs)
        return {k:v.item() for k, v in zip(
            ['rating', 'env', 'flavor', 'service'], outputs.squeeze()
        )}

if __name__ == "__main__":
    predictor = RatingPredictor()
    print("请输入评论：", end='', flush=True)  # 修改提示方式
    try:
        comment = sys.stdin.buffer.readline().strip().decode('utf-8')
    except UnicodeDecodeError:
        comment = sys.stdin.buffer.readline().strip().decode('gbk')
    print(predictor.predict(comment))