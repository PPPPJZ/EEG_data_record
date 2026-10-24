# self_learning 自主运动想象反馈

在项目根目录运行：

```powershell
conda activate EEG_hardware
python -m pip install -r requirements.txt
python app.py
```

浏览器打开 http://127.0.0.1:8000，选择 `self_learning`，填写受试者信息、
学习 trial 总数（默认 100）、teacher `.pt` 文件的绝对路径及空的数据目录。
本阶段接收厂家 WebSocket EEG，不连接震动手环。默认 teacher 设备为 `cuda:0`；
CPU 调试可用 `python app.py --teacher-device cpu`。

真实自主学习必须先在根目录 `config.py` 设置 `TEACHER_CHANNELS`，
或用 `--teacher-channels` 指定 8 个从 1 开始、逗号分隔的采集通道号。
顺序必须对应 Teacher 训练的电极，不能自动取前 8 通道。配置只改变模型输入，
原始 EEG 仍保存全部通道。映射写入 H5 `protocol/json` 的
`acquisition.teacher_channels`，续采必须一致。

设备检查页显示全部采集通道（默认 32，可滚动）。右上模型指示灯只有在 teacher 加载及预热成功
后变绿；模型错误或无有效 EEG 信号时禁止开始。点击模型测试，会逐条处理
10 个随机 `(750,8)` 原始微伏窗口，报告平均和最大耗时以及是否低于推理间隔。
计时包含 40 Hz FIR、标准化、CPU/GPU 传输、ADFCNN、softmax 和返回概率；
排除窗口生成和模型加载。这只检查运行和速度，不检查分类准确率。

光标预览的启动、停止、重置按钮可单独检查脑控效果，不保存 EEG。
预览关闭后可以重复测速；正式开始时自动停止预览。

点击开始采集后，5 秒准备阶段不记录 EEG。每个 trial：

| 时间 | 内容 |
|---|---|
| 0–3 s | 休息，提前自行决定想象哪一侧，显示剩余 trial 数 |
| 3–6 s | 蓝色小球居中，积累本次运动想象的 3 秒原始窗口 |
| 6–13 s | 最近 750 点滑窗反馈，默认每 40 ms 推理一次 |

每次想象开始重置小球到 0.5，同时重置缓冲、EMA 和速度。推理不跨 trial，
不拼接组间休息两侧的数据；过载时不堆积旧窗口。设备断连会终止正式 trial 并
删除当前未完成 trial、保留完整 trial；组间断连暂停，恢复后手动继续。Esc 中止实验并保存已有数据。
组间休息、波形显示及空目录保护沿用原流程。

ADFCNN 沿用 `adfcnn.adfcnn_lib.inference.Predictor`。输入原始 `(750,8)` 微伏，
先取窗口，再按 teacher 内参数做窗口独立的 40 Hz 零相位 FIR 和逐通道标准化。
不使用数采显示的 50 Hz 陷波与两级 5–30 Hz 带通信号；不改变已验证的模型代码。
模型输出右手概率 `p`，控制信号 `s=2p-1`，负值左移、正值右移。
teacher 是固定反馈模型，没有在线更新。无 instruction 时，受试者真实选择未知；
反馈不能证明其想象正确，也不提供准确率评估。

`control.yaml` 的参数在重新确认实验信息后生效：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| inference_interval_ms | 40 | 滑窗推理间隔，毫秒，独立于采集速率 |
| ema_alpha | 0.2 | EMA 新信号权重，`smooth=αs+(1-α)previous` |
| dead_zone | 0.1 | 平滑信号绝对值小于阈值时速度为零，超出部分线性映射到 0–1 |
| max_velocity | 0.2 | 最大速度，屏幕宽度/秒 |
| cursor_min_x | 0.05 | 最左归一化位置 |
| cursor_max_x | 0.95 | 最右归一化位置 |

位置用高精度实际经过时间积分，限制在配置边界内。网页使用
requestAnimationFrame 渲染，依据服务器位置和速度作短暂帧间外推，
推理在线程中执行，不阻塞采集事件循环。固定 250 Hz、750 点和 ADFCNN 架构。
Windows 不保证硬实时，系统负载和厂家数据块到达都会影响更新间隔。
反馈缓冲按块长度容忍正常的推送间隔，建议厂家使用短块推流；
新数据尚未到达时不会重复推理同一窗口。WebSocket 断流检查见根目录 README。

H5 保存全部通道的 raw_uv、events、trials 和 quality，schema 1.4。
新文件不保存 filtered_uv；50 Hz 陷波和两级 5–30 Hz 带通只用于内存中的波形显示。
raw_uv 为厂家接收值的 float32 表示，不额外滤波或乘旧 ADC 系数，输入须为 uV。
厂家上游是否滤波需在厂家程序中确认。
self_learning 的 `trials/table.class_id=-1` 表示未知，不是模型预测标签；
`cue_start_sample` 表示想象控制开始，`cue_end_sample` 与其相同（无箭头）。
新增 `self_learning/config_json` 保存模型哈希、配置与测试结果，
`self_learning/feedback` 保存每次窗口索引、概率、平滑值、速度、位置、
耗时和时间。`control_elapsed_s` 使用高精度时钟，相对当前 trial 控制开始。
所有窗口右边界为 exclusive。`load_session()` 返回 `feedback` 和 `feedback_config`。

文件结构：

```text
control.yaml       六项可调控制参数
control.py         参数验证、原始环形缓冲、EMA/死区/速度积分
runtime.py         teacher 加载预热、模型测试、后台滑窗推理
documentation.py   自主学习会话的数据说明生成
```

无设备测试：

```powershell
python app.py --demo --mock-teacher
python app.py --demo
python -B -m unittest discover -s tests -v
```

第一条使用模拟 EEG 与周期性左右概率，界面明确标记模拟 teacher，
其耗时不代表 ADFCNN。第二条模拟 EEG，但必须提供实际 teacher 路径，仍用真实模型。
模拟模式可通过 `--time-scale 0.2` 加速，真实设备禁止时间缩放。
ADFCNN 自身测试请从 `adfcnn` 目录运行 `python -B -m unittest discover -s tests -v`。

2026-10-20 实测：Windows 11、Python 3.11.17、RTX 5060 Laptop GPU、
PyTorch 2.11.0+cu128，使用 Zhitong_Liu/tactile_MI 的实际 teacher 和真实 Motren_E。
10 条随机窗口测速平均 2.71 ms/条，最大 3.44 ms。
一个完整 trial 保存 2500×8 原始及滤波采样点，覆盖率 100%、异常包 0，
产生 100 次反馈；连续推理平均 3.83 ms，P95 5.02 ms，更新间隔中位数
39.89 ms、P95 41.72 ms。首个窗口在运动想象开始后约 3.01 秒返回。
这验证了运行与速度，不验证模型准确率或自主学习效果。
根目录 33 项测试与 ADFCNN 8 项测试均通过，浏览器三种模式及完整自主学习
流程已验证。验证记录使用临时数据目录，没有改写已有受试者文件或模型。
