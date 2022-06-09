# SMP2020微博情绪分类技术评测（SMP2020-EWECT）

# 队伍：scau_EWECT 机构：scau_SIGSDS

## 相关网址

- [代码仓库——码云](https://gitee.com/a2798063/SMP2020)

- [排行网站](http://39.97.118.137/)

- [官网](https://smp2020.aconf.cn/smp.html)

## 评测集结果

- 通用评测集：使用了5种方案的模型进行投票集成

    1. model-0：RoBERTa-wwm-ext-large(UER) 全连接微调 合并疫情与通用训练集 最大长度150 5折交叉验证集成 代码位于model-0目录下
    
    2. model-1：RoBERTa-wwm-ext-large(UER) 全连接微调 合并疫情与通用训练集 最大长度300 损失函数LMCL 5折交叉验证集成 代码位于model-1目录下

    3. model-2：RoBERTa-wwm-ext-large(UER) 全连接微调 合并疫情与通用训练集 去重+疫情sad、fear、surprise过采样 最大长度300 5折交叉验证集成 代码位于model-2目录下
    
    4. model-3: RoBERTa-wwm-ext-large + BiLSTM 最大长度250 B2*8 合并疫情与通用训练集 5折交叉验证集成 代码位于model-3目录下
    
    5. model-4：RoBERTa-wwm-ext-large(UER) 全连接微调 合并疫情与通用训练集 最大长度300 图片表情转换文字词向量嵌入 5折交叉验证集成 代码位于model-4目录下

- 疫情评测集：使用了3种方案的模型进行投票集成

    1. model-3: RoBERTa-wwm-ext-large + BiLSTM 最大长度250 合并疫情与通用训练集 5折交叉验证集成 代码位于model-3目录下
    
    2. model-5：RoBERTa-wwm-ext-large(UER) 全连接微调 合并疫情与通用训练集 去重+疫情sad、fear、surprise过采样 最大长度300 5折交叉验证集成 代码位于model-5目录下
    
    3. model-6：RoBERTa-wwm-ext-large(UER) + BiLSTM 最大长度250 合并疫情与通用训练集 5折交叉验证集成 代码位于model-6目录下

## 文件说明

- data_split 数据切分代码

- ensemble_voting 投票集成代码以及各个方案预测的结果。文件名后的数字代表相应的模型。

- model-0~6
    
    - data 已切分的5折训练数据，合并疫情和通用训练集，标签转换字典label_dict.json，表情文字转换字典emoji_dict.json，修改的BERT词表vocab.txt

    - data_utils.py 数据加载工具
     
    - langconv.py zh_wiki.py 繁简体转换
    
    - model.py 模型定义文件
    
    - train.py 训练模型，预测
    
    - log.py 日志

- pytorch_chinese_L-24_H-1024_A-16 预训练RoBERTa-wwm-ext-large模型，下载地址：[https://github.com/dbiir/UER-py](https://github.com/dbiir/UER-py)

- pytorch_robert_ext_wwm_large 预训练RoBERTa-wwm-ext-large模型，下载地址：[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

- raw 原始数据

- rules 标签修正规则

- SMP2020微博情绪分类.xlsx  各方案分数记录
