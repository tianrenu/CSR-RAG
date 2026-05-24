# 真实 RAG QA 小规模评测摘要

本评测使用 embedding top-5 检索和 MiniMax 回答，只作为真实 RAG 主实验前的小规模 QA 验证，不作为全量最终结论。

- 样本数：100
- sufficient / insufficient：95 / 5
- Naive RAG EM / F1：0.5300 / 0.7183
- CSR-RAG coverage：0.9900
- CSR-RAG answered EM / F1：0.5354 / 0.7189
- Insufficient answer rate：1.0000
- LLM JSON parse failure rate：0.0000
- LLM strict format failure rate：0.0000

解释时需要谨慎：如果 CSR-RAG 的 answered EM/F1 高于 Naive RAG，但 coverage 较低，说明选择性回答提高了回答子集可靠性；如果没有提升，应把问题归因拆开看，包括检索召回、风险阈值和 LLM 短答案稳定性。
