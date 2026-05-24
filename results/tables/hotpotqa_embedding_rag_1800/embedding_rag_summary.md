# Embedding RAG 实验结果摘要

## 当前定位

这一轮是 CSR-RAG 的真实检索分布实验：检索由 `text-embedding-v4` 的 embedding top-5 决定，风险模型只在 embedding retrieval 的 train split 上训练，calibration 和阈值选择只使用 valid split，test split 只用于最终报告。

## 主结果

- Test sufficient / insufficient: 245 / 25
- Naive RAG decision accuracy: 0.9074
- Uncalibrated CSR decision accuracy: 0.9000, coverage: 0.9926, selective accuracy: 0.9067
- CSR-RAG decision accuracy: 0.8963, coverage: 0.9889, selective accuracy: 0.9064
- CSR-RAG Brier / ECE: 0.0830 / 0.0544

## 解释

如果 CSR-RAG 高于 Naive RAG，论文主实验可以主张：真实 embedding 检索下，轻量 sufficiency risk model 能改善 answer/abstain 决策可靠性。如果提升有限，则应把结论收敛为：controlled sufficiency modeling 有效，但真实 RAG 的最终收益受到 embedding retriever 召回率和 LLM 答案行为共同限制。

当前模型对比中 test decision accuracy 最高的是 `gradient_boosting`，但主方法仍固定为 LogisticRegression + isotonic，非线性模型只作为增强变体讨论。
