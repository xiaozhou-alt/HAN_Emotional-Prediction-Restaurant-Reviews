import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_hidden_dim, sentence_hidden_dim, dropout):
        super().__init__()
        
        # Word-level Attention
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.word_gru = nn.GRU(embed_dim, word_hidden_dim, bidirectional=True, batch_first=True)
        self.word_fc = nn.Linear(2*word_hidden_dim, 2*word_hidden_dim)
        self.word_context = nn.Linear(2*word_hidden_dim, 1, bias=False)
        
        # Sentence-level Attention
        self.sentence_gru = nn.GRU(2*word_hidden_dim, sentence_hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_fc = nn.Linear(2*sentence_hidden_dim, 2*sentence_hidden_dim)
        self.sentence_context = nn.Linear(2*sentence_hidden_dim, 1, bias=False)
        
        # Regression Heads
        self.regression = nn.ModuleList([nn.Linear(2*sentence_hidden_dim, 1) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, sentences, words)
        batch_size = x.size(0)
        
        # Word-level processing
        x = self.word_embed(x)  # (batch, sent, word, emb)
        x = self.dropout(x)
        
        # Process each sentence
        word_attentions = []
        for sent_idx in range(x.size(1)):
            sentence = x[:, sent_idx, :, :]  # (batch, words, emb)
            word_out, _ = self.word_gru(sentence)  # (batch, words, 2*hidden)
            
            # Word attention
            word_energy = torch.tanh(self.word_fc(word_out))
            word_energy = self.word_context(word_energy).squeeze(-1)  # (batch, words)
            word_alpha = F.softmax(word_energy, dim=1).unsqueeze(1)  # (batch, 1, words)
            
            sentence_embed = torch.bmm(word_alpha, word_out).squeeze(1)  # (batch, 2*hidden)
            word_attentions.append(sentence_embed)
        
        # Sentence-level processing
        sentences = torch.stack(word_attentions, dim=1)  # (batch, sent, 2*hidden)
        sent_out, _ = self.sentence_gru(sentences)  # (batch, sent, 2*hidden)
        
        # Sentence attention
        sent_energy = torch.tanh(self.sentence_fc(sent_out))
        sent_energy = self.sentence_context(sent_energy).squeeze(-1)  # (batch, sent)
        sent_alpha = F.softmax(sent_energy, dim=1).unsqueeze(1)  # (batch, 1, sent)
        
        document_embed = torch.bmm(sent_alpha, sent_out).squeeze(1)  # (batch, 2*hidden)
        
        # Regression outputs
        outputs = [head(document_embed) for head in self.regression]
        return torch.cat(outputs, dim=1)