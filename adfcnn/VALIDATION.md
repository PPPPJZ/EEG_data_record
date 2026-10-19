# 验证记录

验证日期：2026-10-19。执行环境为 Linux、Python 3.11.15、NVIDIA RTX 6000D、PyTorch 2.11.0+cu128，依赖版本与 requirements.txt 一致。未在 Windows 实机执行；推荐的 PyTorch wheel 已确认包含 Python 3.11 的 Windows 64 位版本。

## 实际 H5 完整训练

将整个 adfcnn 文件夹及示例 motor_imagery.h5 复制到临时目录，从 /tmp 调用复制后脚本，训练代码不访问 A-EEG 其他目录。

输入：Haoyu_Xu / pure_MI_1 / motor_imagery.h5。80 个计划 trial、52 个已完成且有完整 cue 后 750 点窗口的有效 trial，左手 27、右手 25；排除 28 个 trial。每个 seed 的五折大小均为 11、11、10、10、10。

| 检查 | 结果 |
|---|---|
| results：3 个 seed × 5 折 × 300 epochs | 全部 15 次完成；平均 Test Accuracy 47.45% |
| teacher：seed 42、单次 300 epochs | Test Accuracy 45.45%；最佳 epoch 15 |
| teacher 划分 | train 30 / val 11 / test 11 |
| teacher 模型文件 | 保存到临时输入 H5 所在目录；包含权重与推理配置 |
| 独立 reasoning 命令行 | 原始 (750, 8) .npy 窗口，GPU 与 CPU 均加载成功 |
| GPU 与 CPU 一致性 | 预测类别相同，两类概率最大差异约 6e-8 |
| results 文件写入 | 代码与输入 H5 哈希未变，未创建 checkpoint、缓存或字节码 |
| 模型架构对齐 | 相同权重及 (2, 1, 8, 750) 输入下，输出与参考实现逐值相同 |

上述 Accuracy 是代码验证时此示例数据的结果，不代表其他受试者或未来采集数据的准确率。验证使用的模型、数据副本和日志均在临时目录，未放入代码文件夹。

## 自动化检查

8 项测试覆盖：trial/标签/shuffle 对齐、训练与 Feedback 单窗预处理一致、分层五折互斥及完整覆盖、无效窗口与 NaN/Inf 拒绝、通道顺序错误拒绝、每类 trial 不足拒绝、验证准确率优先及 loss 决胜和 test 单次评估、teacher 保存/加载/重复推理与不兼容版本拒绝。

代码语法检查通过；requirements.txt 中各版本与执行环境及其直接依赖约束一致。

```powershell
python -B -m unittest discover -s tests -v
```
