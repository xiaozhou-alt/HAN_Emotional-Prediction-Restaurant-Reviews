import pandas as pd
import jieba
import pickle
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from collections import Counter
import csv
from io import StringIO
from tqdm import tqdm
import sys
import gc

def split_sentences(text):
    """分句函数优化版"""
    delimiters = {'。', '！', '？', '；', '，', '…','~'}
    sentences = []
    buffer = []
    for char in text:
        buffer.append(char)
        if char in delimiters:
            sentences.append(''.join(buffer).strip())
            buffer = []
    if buffer:
        sentences.append(''.join(buffer).strip())
    return sentences

def preprocess(config):
    # 解除CSV字段大小限制
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)

    # 一次性加载全部数据
    print("正在加载和清洗数据...")
    with open(config['data']['raw_path'], 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()[1:]  # 跳过标题行
    
    # 内存优化：限制处理行数为200万
    raw_lines = raw_lines[:500000]

    # 数据清洗管道
    cleaned_lines = []
    for line in tqdm(raw_lines, desc="清洗数据行"):
        line = line.strip() \
            .replace('"', '') \
            .replace(',', '\t')  # 统一分隔符
        
        # 列数据验证增强
        parts = line.split('\t')[:8]  # 强制截断到8列
        if len(parts) < 8:
            parts += [''] * (8 - len(parts))
        
        # 数值列验证（第3-6列为评分）
        for i in range(2,6):
            try:
                parts[i] = str(float(parts[i]))
            except:
                parts[i] = ''
        cleaned_lines.append('\t'.join(parts[:8]))

    # 创建DataFrame
    column_names = [
        "userId", "restId", "rating", "rating_env",
        "rating_flavor", "rating_service", "timestamp", "comment"
    ]
    df = pd.read_csv(
        StringIO('\n'.join(cleaned_lines)),
        sep='\t',
        header=None,
        names=column_names,
        engine='python',
        quoting=csv.QUOTE_NONE,
        dtype={'comment': str},
        error_bad_lines=False,
        warn_bad_lines=True,
        na_values=['', 'NA','null']
    )

    # 内存优化：立即释放不再使用的变量
    del raw_lines, cleaned_lines
    gc.collect()

    # 评分处理增强
    print(f"\n原始数据量: {len(df)}条")
    rating_columns = ['rating', 'rating_env', 'rating_flavor', 'rating_service']
    
    # 过滤空值并转换数值类型
    df_cleaned = df.dropna(subset=rating_columns, how='all')
    for col in rating_columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=rating_columns, how='any')
    
    # 评分范围验证
    df_cleaned = df_cleaned[
        df_cleaned[rating_columns].apply(
            lambda x: x.between(1,5).all() & x.notnull().all(),
            axis=1
        )
    ]
    
    # 文本处理管道
    df_cleaned['comment'] = df_cleaned['comment'].fillna('')
    tqdm.pandas(desc="处理评论内容")
    df_cleaned['comment'] = df_cleaned['comment'].progress_apply(
        lambda x: re.sub(r'\s+', ' ', x.replace('\n',' ')).strip()
    )
    
    print(f"有效数据量: {len(df_cleaned)}条")
    print(f"空白评论数量: {df_cleaned['comment'].str.strip().eq('').sum()}条")

    # 构建数据集
    comments = df_cleaned['comment'].tolist()
    ratings = df_cleaned[rating_columns].to_numpy(dtype=np.float32)

    # 分词处理优化
    word_counter = Counter()
    processed_docs = []
    with tqdm(total=len(comments), desc="处理评论") as pbar:
        for comment in comments:
            if not isinstance(comment, str) or not comment.strip():
                processed_docs.append([[1]])  # <UNK>标记
                pbar.update(1)
                continue
                
            sentences = split_sentences(comment)[:config['data']['max_sentences']]
            doc = []
            for sent in sentences:
                # 分词优化：过滤非中文字符
                words = [
                    w.strip() for w in jieba.cut(sent) 
                    if w.strip() and re.match(r'[\u4e00-\u9fa5]', w)
                ][:config['data']['max_words']]
                if words:
                    word_counter.update(words)
                    doc.append(words)
            processed_docs.append(doc or [[1]])
            pbar.update(1)

    # 生成词汇表
    print("\n生成词汇表...")
    vocab = ['<PAD>', '<UNK>'] + [w for w,_ in word_counter.most_common(config['data']['vocab_size']-2)]
    word2idx = {w:i for i,w in enumerate(vocab)}

    # 数据编码
    final_data = []
    for doc in tqdm(processed_docs, desc="编码文档"):
        encoded_doc = []
        for sent in doc[:config['data']['max_sentences']]:
            indices = [word2idx.get(w,1) for w in sent][:config['data']['max_words']]
            indices += [0]*(config['data']['max_words']-len(indices))
            encoded_doc.append(indices)
        # 填充空句子
        encoded_doc += [[0]*config['data']['max_words']]*(config['data']['max_sentences']-len(encoded_doc))
        final_data.append(encoded_doc)

    # 数据集分割
    print("\n保存预处理结果...")
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        final_data, ratings, 
        test_size=config['training']['test_size'], 
        random_state=42
    )

    # 保存结果
    with open(os.path.join(config['data']['processed_dir'], 'train.pkl'), 'wb') as f:
        pickle.dump({'X':X_train, 'y':y_train}, f)
    with open(os.path.join(config['data']['processed_dir'], 'test.pkl'), 'wb') as f:
        pickle.dump({'X':X_test, 'y':y_test}, f)
    with open(os.path.join(config['data']['processed_dir'], 'vocab.pkl'), 'wb') as f:
        pickle.dump(word2idx, f)

if __name__ == "__main__":
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    preprocess(config)