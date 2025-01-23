# 分词技术 (Tokenization)

分词是大语言模型处理文本的第一步，其质量直接影响模型的性能。本文档介绍主流的分词算法及其实现。

## 1. 分词概述

### 1.1 基本概念
- Token：文本的基本单位
- Vocabulary：词表，包含所有可用的 token
- OOV (Out-of-Vocabulary)：词表外词

### 1.2 分词的重要性
- 影响模型的词表大小
- 决定了信息的粒度
- 影响模型处理多语言的能力
- 影响压缩率和效率

## 2. BPE (Byte Pair Encoding)

### 2.1 算法原理
1. 从字符级别开始
2. 统计相邻字符对的频率
3. 合并最常见的字符对
4. 重复直到达到目标词表大小

### 2.2 实现示例
```python
def learn_bpe(text, num_merges):
    # 初始化词表为字符级别
    vocab = {c: i for i, c in enumerate(set(text))}
    
    # 统计字符对频率
    def get_stats(vocab):
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    # 合并最频繁的字符对
    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in v_in:
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = v_in[word]
        return v_out
    
    # 主循环
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    
    return vocab
```

## 3. WordPiece

### 3.1 原理
- 类似BPE，但使用不同的评分机制
- 基于概率而不是频率
- 常用于BERT等模型

### 3.2 特点
- 倾向于保留完整词
- 更好的语言学意义
- 处理未知词的策略

## 4. SentencePiece

### 4.1 特性
- 语言无关的分词器
- 将空格也作为基本单位处理
- 支持多种分词算法

### 4.2 使用示例
```python
import sentencepiece as spm

def train_sentencepiece(input_file, vocab_size, model_type='bpe'):
    # 训练模型
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix='spm',
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        max_sentence_length=4192
    )
    
    # 加载模型
    sp = spm.SentencePieceProcessor()
    sp.load('spm.model')
    
    return sp

# 使用示例
def tokenize_text(sp, text):
    # 编码
    tokens = sp.encode_as_pieces(text)
    # 解码
    decoded = sp.decode_pieces(tokens)
    return tokens, decoded
```

## 5. 实践考虑

### 5.1 词表大小选择
- 太小：信息损失
- 太大：训练困难
- 推荐范围：8k-50k

### 5.2 特殊token
- [PAD]：填充标记
- [UNK]：未知词标记
- [CLS]：分类标记
- [SEP]：分隔标记
- [MASK]：掩码标记

### 5.3 多语言支持
- Unicode规范化
- 字符集覆盖
- 语言特定处理

## 6. 评估指标

### 6.1 定量指标
- 词表大小
- 压缩率
- OOV率
- 分词速度

### 6.2 定性分析
- 语言学合理性
- 跨语言能力
- 特殊字符处理
- 鲁棒性

## 7. 最佳实践

### 7.1 训练建议
- 使用大规模语料
- 合适的词表大小
- 充分的字符覆盖率
- 处理特殊情况

### 7.2 使用建议
- 规范化预处理
- 合理的默认值
- 错误处理机制
- 性能优化

## 8. 参考资源

### 8.1 论文
1. BPE原始论文
2. WordPiece相关研究
3. SentencePiece论文

### 8.2 工具
- HuggingFace Tokenizers
- SentencePiece
- 其他开源实现

### 8.3 教程和示例
- 官方文档
- 实践指南
- 示例代码