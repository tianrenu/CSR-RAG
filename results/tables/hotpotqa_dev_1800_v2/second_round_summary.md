# CSR-RAG 第二轮实验结果摘要

## 1. 实验定位

第二轮实验用于把第一轮 credible prototype 推进为 paper-ready evidence。当前仍只研究检索充分性风险建模，不接最终 QA 生成，也不扩展 refine / clarify。

## 2. 主结果

- CSR-RAG 主方法：LogisticRegression + isotonic calibration + all_features
- test decision accuracy：0.5815
- test coverage：0.5630
- test selective accuracy：0.5724
- calibrated Brier：0.2402
- calibrated ECE：0.0369

## 3. 与未校准 CSR 的关系

- Uncalibrated CSR decision accuracy：0.5759
- CSR-RAG decision accuracy：0.5815
- raw ECE：0.0244
- calibrated ECE：0.0369

如果 ECE 没有全面下降，论文表述应收敛为：calibration changes the coverage-risk trade-off，而不是 calibration always improves score quality。

## 4. 模型与特征结论

- isotonic 下表现最好的 estimator：logistic_regression，decision accuracy = 0.5815
- LR 消融中表现最好的 feature set：retrieval_only，decision accuracy = 0.5833

非线性模型如果强于 LogisticRegression，应在论文中作为 stronger estimator variant，而不是替换 CSR-RAG 主线。主线仍然保持轻量、可解释、可复现。

## 5. 当前能支撑的论文主张

- 检索充分性预测比 always-answer 更可靠。
- 不同 estimator 和 feature group 会明显影响选择性回答效果。
- 校准模块影响 coverage 与 selective accuracy 的取舍。
- 当前实验仍不能声称最终 QA 生成质量提升，因为尚未接生成评测。
