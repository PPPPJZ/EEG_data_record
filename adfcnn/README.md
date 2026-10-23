# ADFCNN：Windows 训练与 Feedback 推理

本目录是 `EEG_data_record` 的 ADFCNN 模块。模型、H5 解析、预处理、训练和推理代码均在本目录内，不导入项目的其他 Python 模块。Python 环境由项目根目录的 `environment.yml` 和 `requirements.txt` 统一管理。

## Windows 环境

推荐 64 位 Windows 10/11、Python 3.11、NVIDIA GPU，以及支持 CUDA 12.8 的 NVIDIA 驱动。在 Anaconda Prompt 或已初始化 conda 的 PowerShell 中执行：

```powershell
cd "D:\your_project\EEG_data_record"
conda env create -f environment.yml
conda activate EEG_hardware
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
cd adfcnn
```

根目录 `requirements.txt` 固定 PyTorch 2.11.0 + CUDA 12.8，并指定官方 CUDA wheel 下载源，避免在 Windows/Linux 上装成 CPU 版。PyTorch wheel 已包含所需 CUDA 运行时，无需安装 CUDA Toolkit。最后一条检查命令的 `GPU available` 应为 `True`。环境创建后，三个入口都使用 `cuda:0`，可通过 `--device cuda:1` 选择其他 GPU。

Linux 使用同样的 Python 环境和安装命令。迁移到另一台机器时重新创建环境、安装依赖，复制项目和 teacher 文件即可；不需要复制服务器上的 conda 目录。训练流程已在 Linux Python 3.11 + NVIDIA GPU 上验证，推理流程已在 Windows Python 3.11 + NVIDIA GPU 上验证。

## results：五折实验，只打印平均准确率

```powershell
python results.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5"
```

协议与 `EEG_data/adfcnn_test_10_23/RESULTS.md` 一致。默认训练 5 次，每次 2000 epochs。stdout 最终只打印一行：

```text
Accuracy: xx.xx%
```

读取后先用固定 seed `42` 同步打乱 trial、标签和 trial ID。随后使用 seed `42` 只建立一次分层五折，每折左右手 trial 数必须严格相等。test 依次为 fold 1–5，其余四折训练；没有 validation。所有 trial 各做一次 test。每类至少需要 5 个有效 trial；若分层划分不能满足每折严格平衡，则明确报错，不删 trial、不复制样本。

Accuracy 是 5 次 **Test Accuracy 的算术平均值**。每折都训练满 2000 epochs，再使用最终模型在 test 上评估一次，不选择 checkpoint。每次实验都用 seed `42` 重新初始化模型、优化器和学习率调度器。

results 不保存 checkpoint、日志、缓存、CSV、图表或 Python 字节码。执行期间如需要看到进度：

```powershell
python results.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5" --verbose
```

`--verbose` 将数据统计、每 25 epochs 的训练 loss/学习率、各折 test Accuracy 输出到 stderr，stdout 仍只输出最后平均 Accuracy。不写日志文件。

## teacher：全量训练，保存个人模型

```powershell
python teacher.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5"
```

teacher 固定训练 seed `42`，使用输入 H5 的全部有效带真实左右手标签 trial 训练一次，训练满 2000 epochs。没有 folds、validation、test、欠采样或 checkpoint 选择，保存最后一个 epoch 的权重。左右手数量可不相等，每类有至少 1 个有效 trial 即可。默认仅输出模型路径：

```text
Model: D:\data_record\Haoyu_Xu\pure_MI_1\adfcnn_teacher.pt
```

`adfcnn_teacher.pt` 默认保存到实际输入 H5 的所在目录，不使用 H5 metadata 中原采集机器的 `output_dir`。再次运行会原子替换同名模型。需要保留已有模型或改变位置时指定输出文件：

```powershell
python teacher.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5" --output "D:\models\Haoyu_Xu_v2.pt" --verbose
```

模型文件含 CPU 权重、网络配置、完整预处理参数、通道/类别约定、最终 epoch、最终训练 loss/学习率、训练 seed、全部训练 trial ID、实际类别计数、全量训练协议和软件版本。由于没有留出评估集，模型文件不记录 val/test Accuracy。`Predictor` 使用权重及配置恢复推理流程，加载时不依赖训练 H5、原始数据路径或训练机器的 GPU 编号；新旧 teacher 文件均可用于现有推理入口。

## reasoning：原始 3 秒窗口分类

输入必须为 **原始微伏 EEG**，形状严格为 `(750, 8)`，250 Hz，列顺序 `CH1–CH8`，包含完整 cue 后 3 秒数据。不要传入软件已经滤波、裁剪、标准化的信号，不要转置，也不要包含标签列。

分类 ID：`0=left_hand`（左手），`1=right_hand`（右手）。

Feedback 集成时，在程序启动时加载一次模型，每收到一个完整窗口调用一次：

```python
import numpy as np
from reasoning import Predictor

predictor = Predictor(r"D:\data_record\Haoyu_Xu\pure_MI_1\adfcnn_teacher.pt", device="cuda:0")

# 在开始采集前预热一次，避免首个真实 Feedback 承担库初始化开销。
predictor.predict(np.zeros((750, 8), dtype=np.float32))

# raw_window_uv 由实际采集程序提供，shape=(750, 8)，单位 uV。
raw_window_uv = np.asarray(raw_window_uv, dtype=np.float32)
class_id = predictor.predict(raw_window_uv)

# 如果 Feedback 还需要类别名称和概率：
result = predictor.predict_details(raw_window_uv)
print(result.class_id, result.class_name, result.probabilities)
# probabilities 的顺序固定为 (left_hand, right_hand)。
```

当前 Windows 11、Python 3.11.17、RTX 5060 Laptop GPU、PyTorch 2.11.0+cu128 环境下，使用真实 H5 单窗预热后重复 500 次，`predict_details()` 端到端中位数为 **2.44 ms**，P95 为 **2.58 ms**；其中模型前向中位数为 1.41 ms，其余主要为 FIR 预处理和数据传输。模型加载不计入上述时间。新进程中，模型加载后的首次调用因 MNE/PyTorch 延迟初始化约为 2.32 s，因此实时采集前必须完成上面的预热；具体时延会随硬件、驱动和系统负载变化。

上例调用方可在 `adfcnn` 中运行；若外部程序与 `adfcnn` 同级，也可 `from adfcnn.adfcnn_lib.inference import Predictor`。无需修改库里的路径。`predict` 每次返回 Python `int`；概率为 softmax 输出，未做置信度校准。

命令行读取 `.npy`：

```powershell
python reasoning.py --model "D:\data_record\Haoyu_Xu\pure_MI_1\adfcnn_teacher.pt" --window "D:\windows\raw_3s.npy"
```

默认仅输出 `0` 或 `1`。附加 `--details` 则输出一行 JSON：

```json
{"class_id": 0, "class_name": "left_hand", "probabilities": [0.7, 0.3]}
```

## 信号处理和训练协议

H5 必须包含 `metadata/json`、`protocol/json`、`eeg/raw_uv`、`trials/table`。直接读取 trial 的 `class_id`、`completed`、`cue_start_sample`、`trial_end_sample`；不根据预定执行序列推断标签或时间。

1. 仅保留已完成、类别为 0/1、边界有效、含完整 750 点窗口的 trial。未完成文件也可训练，只使用有效 trial。results 每类至少需要 5 个，并要求每折左右手数量严格相等；teacher 使用全部有效 trial，每类至少需要 1 个。`class_id=-1` 的 self_learning trial 没有真实标签，不能用于此监督训练。有效窗口含 NaN/Inf 则报错，不静默补零。
2. 从 `cue_start_sample` 取左闭右开切片 `[start:start+750]`。3 秒按 750 点定义，不增加第 751 点，不重采样。
3. 每个窗口独立进行 40 Hz 零相位 FIR 低通（MNE，Hamming/firwin，自动滤波长度和过渡带，边缘使用 reflect\_limited）。训练和推理调用同一函数，不使用窗口之外的数据，不跨 trial 或组间休息滤波。
4. 每个窗口、每个通道独立标准化：`(x - mean) / max(std, 1e-4)`，std 使用 `ddof=0`，输出 float32 `(1, 8, 750)`。不跨 trial 计算均值/方差，不在全部数据上拟合 scaler。
5. ADFCNN 保留参考实现的双分支和注意力计算：8 个特征，dropout 0.25，卷积 max norm 2.0；两路池化合计 51 个位置，分类器输出两类 log-softmax。
6. 两个入口独立显式配置相同超参数：训练 2000 epochs，batch size 16，Adam，学习率 0.001，weight decay 0.075；CosineAnnealingLR，`T_max=epochs`（默认 2000）、`eta_min=0`，CrossEntropyLoss。训练 DataLoader 每个 epoch shuffle，保留最后不足一个 batch 的样本。
7. results 使用平衡五折，每轮四折训练、一折测试，最后一个 epoch 后测试一次。teacher 使用全部有效带标签数据训练，保存最终模型，整个过程不调用评估函数。两个入口都没有 early stopping 或 validation。

旧 inner-session 代码曾使用 751 点窗口，并存在连续 block 滤波路径。本实现按照本次确认使用 750 点和窗口独立滤波，因此保证训练与实时窗口推理一致；其结果不应被视为旧预处理结果的逐数值复现。

划分 seed 不依赖文件名或路径。results 的划分和每折训练、teacher 的全量训练都使用固定 seed `42`，不按 fold 编号偏移。固定 Python、NumPy、PyTorch 随机种子及确定性算法。同一软件和硬件环境下可重复；不同 GPU 或依赖版本之间不保证浮点结果逐位一致。

## 参数位置与快速检查

`results.py` 和 `teacher.py` 顶部各有独立的参数配置区。读取 seed、results seed、teacher seed 和五折数在 `adfcnn_lib/config.py` 明确列出。命令行支持 `--device`、`--epochs`、`--batch-size`；正式实验使用默认 2000 epochs。

仅检查程序能否跑通时，可以减少 epochs：

```powershell
python results.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5" --epochs 1 --verbose
python teacher.py --data "D:\data_record\Haoyu_Xu\pure_MI_1\motor_imagery.h5" --epochs 1 --output "D:\temp\smoke_teacher.pt"
python -B -m unittest discover -s tests -v
```

无 GPU 时可显式指定 `--device cpu` 做调试；默认不会自动退回 CPU。

目录中的 `tests` 使用临时合成 H5 验证 trial 与标签对齐、严格平衡的 4/1 划分、最终模型仅测试一次、teacher 全量训练且不评估、预处理一致性，以及模型保存/加载；不把测试输出写到数据目录。
