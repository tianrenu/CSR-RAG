# CSR-RAG 第三轮实验结果摘要

## 1. 当前最好结果

- 主方法：LogisticRegression + isotonic + feature v2
- decision accuracy：0.6593
- coverage：0.4037
- selective accuracy：0.6972
- calibrated Brier：0.2092
- calibrated ECE：0.0375

## 2. feature v1 vs v2

- v1 decision accuracy：0.5815
- v2 decision accuracy：0.6593
- v1 calibrated Brier：0.2402
- v2 calibrated Brier：0.2092

## 3. 模型、特征与校准

- 最好 estimator：logistic_regression，decision accuracy = 0.6593
- 最好特征组设置：no_query，decision accuracy = 0.6815
- 未校准 CSR decision accuracy：0.6574
- 校准后 CSR-RAG decision accuracy：0.6593

## 4. 论文表述建议

- 继续把主张收敛在检索充分性风险预测和选择性回答。
- 如果 feature v2 未明显提升，就不继续堆特征，应转向论文写作和局限性分析。
- 如果 feature v2 明显提升，可以进入最小 QA generation 评测，但仍保持 answer / abstain 主线。
