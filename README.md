# 运动想象 EEG 数据采集：pure_MI / tactile_MI / self_learning

Python 按厂家 `ws_server_example.py` 协议接收 WebSocket EEG 推流，
默认 **32 通道、250 Hz**。浏览器用于实验设置、全部通道的实时波形检查、
运动想象提示和 teacher 光标反馈。支持三种采集模式：

| 模式 | 刺激 |
|---|---|
| `pure_MI` | 纯运动想象，沿用原有流程 |
| `tactile_MI` | 在运动想象期间给同侧手腕施加高低交替震动 |
| `self_learning` | 受试者自主选侧，用固定 ADFCNN teacher 控制蓝色小球 |

本目录是可独立运行的发布目录，不依赖 `EEG_data_test` 或其他项目目录。

## 快速启动：已有 EEG_hardware 环境

在终端中运行：

```bash
conda activate EEG_hardware
cd C:/Users/WAIS/PPPJZ/EEG_data_record
python -m pip install -r requirements.txt
python app.py
```

在其他电脑上，把 `cd` 的路径换成下载后的项目目录；Windows 的 conda 命令
可在 Miniconda Prompt 或已初始化 conda 的 PowerShell 中执行。

浏览器访问 **http://127.0.0.1:8000**。选择模式、确认受试者信息后，
本程序开始监听 **ws://127.0.0.1:8765**；在厂家程序中连接新 EEG 设备并启用
向该地址转发。收到有效 EEG 后显示连接成功，可按原流程开始实验。
`tactile_MI` 仍需打开蓝牙及 `HapticL`、`HapticR`。

### 厂家推流接入

本程序已经承担示例的 WebSocket 服务器角色，**不用同时运行
`ws_server_example.py`**，否则会占用同一端口。厂家程序负责硬件连接、启动
采样及发送，本程序不向新设备发送旧 BLE 的开始/停止命令。

接收 JSON 格式：

```json
{"type": "eeg_data", "data": [[1.0, 2.0], [3.0, 4.0]], "sample_rate": 250}
```

上述只展示 2 通道 × 2 点结构；默认实际需要 32 行，每行是一段连续采样点，
所有行等长。仅接收有限数值，通道数和 250 Hz 必须匹配，不转置、不补零、
不重采样、不乘 `Motren_E` 的 ADC 系数。推流须使用 **uV**。
新 H5 的 `raw_uv` 是接收值转为 float32，未经过本程序滤波；若厂家上游已
滤波，此处无法还原。因此请在厂家端确认转发的是未滤波的通道数据及其单位。
示例消息没有硬件时间戳或包序号，事件索引沿用原系统“事件发生时已收到的
采样点”规则。厂家缓存与块长度会影响边界对齐精度，本程序无法识别上游
已经丢失的采样；应使用实时短块推流。

地址、通道数、断流超时与 Teacher 通道映射集中在 `config.py` 的
`EEG_WS_*` / `TEACHER_CHANNELS` 和 `EEGStreamConfig`，也可通过参数覆盖：

```bash
python app.py --eeg-ws-host 127.0.0.1 --eeg-ws-port 8765 --eeg-channels 32 --eeg-timeout 3
```

厂家转发目标要与监听地址一致。跨电脑连接时监听可设 `0.0.0.0`，
厂家目标填写本采集电脑的局域网 IP。网页 `--port` 与推流 `--eeg-ws-port`
是两个独立端口。一次只接收一个推流客户端，额外连接被拒绝。
通道顺序以厂家 `data` 的行顺序为准，默认标签 CH1–CH32，未猜测电极位置。
需要实际电极标签时在 `EEGStreamConfig.channel_labels` 中按发送顺序配置。

当前提供的 `config/app_settings.json` 中 `network.port=5001` 属于厂家
设备网络连接配置，不能当作这里的 WebSocket 端口；该文件未见明确的转发
配置项。实际推流需厂家提供支持 `recorder.streamer` 的程序版本及转发设置方式。
本项目的本地协议测试不等于已验证厂家 EXE 能发出数据。

IDE 的 Python 解释器也应选为 `EEG_hardware`。如果 IDE 提示未安装依赖，
先核对解释器是否与终端一致：

```bash
python -c "import sys; print(sys.executable)"
python -m pip check
```

## 新电脑安装

推荐 Python **3.11**，最低 3.10。macOS、Windows、Linux 均使用同一套 Python
代码和 `requirements.txt`，Bleak 自动选择系统原生 BLE 后端。

使用 conda：

```bash
cd EEG_data_record
conda env create -f environment.yml
conda activate EEG_hardware
python app.py
```

已有同名环境时，使用上面的 `pip install -r requirements.txt`，无需重建。
也可以自行创建虚拟环境，然后安装 `requirements.txt`。

`requirements.txt` 显式包含 Pydantic、WebSocket 和 `adfcnn` 所需依赖，不需要
Node.js 或前端构建。Windows/Linux 会安装 PyTorch 2.11.0 + CUDA 12.8 wheel，
其中已包含 CUDA 运行时，无需另装 CUDA Toolkit；ADFCNN 使用 GPU 时仍需兼容
CUDA 12.8 的 NVIDIA 驱动。macOS 安装同版本的 CPU PyTorch。

若希望复用本次 macOS / Python 3.11 下验证的数采直接依赖版本：

```bash
python -m pip install -r requirements.txt -c constraints-tested.txt
```

版本约束不固定平台专用依赖；其他系统仍由 Bleak 自动安装适合的依赖。

### 操作系统准备

| 系统 | 要求 |
|---|---|
| macOS | 开启蓝牙，允许运行 Python 的终端或 IDE 访问蓝牙；权限在“系统设置 → 隐私与安全性 → 蓝牙”中检查 |
| Windows | 推荐 Windows 10/11，使用支持 BLE 的适配器与正常工作的驱动，开启系统蓝牙 |
| Linux | 需要运行 BlueZ（建议 5.55 或以上）、D-Bus 和可访问蓝牙适配器的普通用户会话；开启适配器并解除 rfkill 屏蔽 |

Ubuntu / Debian 的 Linux 系统准备示例：

```bash
sudo apt install bluez
sudo systemctl enable --now bluetooth
bluetoothctl power on
```

马达使用广播名定位，不依赖 macOS 地址或特定电脑路径。关闭其他占用
马达的连接程序；厂家 EEG 程序需保持运行。`tactile_MI` 需要适配器支持同时连接 **2 个 BLE 马达**，
具体能力取决于适配器与驱动。扫描过程串行协调，连接建立后会保持连接。

旧版 BLE 数采和 Teacher 的验证记录不代表新设备已验证。
新版本使用本地 WebSocket 推流测试接收与存储；真实厂家推流、单位和触觉同步仍需实机验证。Python 依赖不能
代替操作系统蓝牙服务、权限或设备驱动。

## self_learning

选择自主学习后，设置总 trial 数（默认 100）和 teacher 模型绝对路径，
不填写左右手 trial 数，也不连接马达。设备检查页右上显示模型加载及预热状态，
“模型测试”对 10 条随机原始窗口逐条推理并报告耗时。
每个 trial 的 0–3 秒自行决定想象方向，3–13 秒显示蓝色小球；想象窗口收齐
750 点后（通常第 6 秒）开始 40 ms 滑窗反馈。每次 trial 重置缓冲和光标。

本阶段标签为未知 `-1`，模型反馈单独保存，不代表受试者真实选择。
模型只用原始 EEG 窗口的 40 Hz FIR 和标准化，与波形显示滤波独立。
现有 Teacher 固定输入 8 通道，需要先按训练时的电极顺序配置
`TEACHER_CHANNELS`，或传入 `--teacher-channels`（8 个从 1 开始的通道号，
以逗号分隔）。映射未配置时真实自主学习会给出错误，不能默认为前 8 通道。
保存和显示仍保留全部通道，映射保存在 `/protocol/json` 的
`acquisition.teacher_channels`。续采会校验映射相同。
离线 ADFCNN 训练器仍要求 8 通道输入，不能直接把 32 通道文件当成旧数据使用。
厂家分块大小决定新窗口的到达频率；建议短块推送以获得及时反馈，40 ms
调度不会创造尚未收到的新采样点。
控制配置在 `self_learning/control.yaml`，详细使用方法见
[自主学习模块说明](self_learning/README.md)。

## pure_MI / tactile_MI 操作流程

1. 首页先选择 `pure_MI`、`tactile_MI` 或 `self_learning`。
2. 填写受试者编号、日期、描述、左右手 trial 数、每 N 个 trial 的休息间隔
   和数据目录。左右手默认各 50 个，N 默认 20；每次按指定数量随机排列。
3. `tactile_MI` 额外显示震动强度滑条：范围 0–1，步长 0.01，默认 **0.8**。
4. 数据目录需使用当前采集电脑上的绝对路径，允许不存在或为空。非空目录
   会被拒绝，不覆盖已有数据。例如：
   - macOS：`/Users/name/EEG_data/S001_tactile`
   - Windows：`C:\EEG_data\S001_tactile`
   - Linux：`/home/name/EEG_data/S001_tactile`
5. 确认信息后才启动 EEG 接收。设备检查页不限时等待，检查全部通道波形（多通道可滚动）；
   `tactile_MI` 同时显示左右马达连接状态，并提供“震动测试”。
6. 点击“震动测试”：两侧以设置的高档强度恒定震动 **2 秒**，之后自动停止。
   可重复测试；测试期间不能开始采集。
7. 点击“开始采集”：先显示 5 秒十字准备页，此阶段不保存 EEG。随后进入
   正式 trial。
8. 每完成 N 个 trial 且还有后续 trial，显示 3 秒闭眼提示，再进入 30 秒
   波形监控休息。剩余 6 秒响铃，倒计时归零后进入下一 trial 的休息阶段。
9. 组间休息可暂停；手动点击继续会从头计时完整 30 秒。`tactile_MI` 此时
   也可测试震动；**测试期间暂缓计时，停止后从剩余时间继续**。原本手动
   暂停的倒计时仍保持暂停。
10. 正式采集或组间休息时按 `Esc` 中断实验，保留已经采集的不完整数据；
    设备检查时按 `Esc` 可以提前停止正在执行的震动测试。

## trial 时序与触觉协议

每个 trial 固定 **8 秒**，第一个 trial 也先休息：

| trial 内时间 | 视觉提示 | pure_MI | tactile_MI |
|---|---|---|---|
| 0–2.5 s | 休息 | 无震动 | 无震动 |
| 2.5–4.0 s | 左/右箭头 cue | 无震动 | 同侧高低交替震动 |
| 4.0–8.0 s | 黑屏运动想象 | 无震动 | 继续同侧高低交替震动 |
| 8.0 s | trial 结束 | — | 立即发出停止命令 |

箭头左侧 → `HapticL`；箭头右侧 → `HapticR`。对侧马达保持停止。
刺激窗口从 cue 开始到 trial 结束，计划持续 **5.5 秒**，与运动想象进度条一致。
准备阶段、trial 休息及组间休息都不施加实验震动；手动测试是单独的 2 秒测试。

高档为滑条设置值，低档为 `max(0, 高档 - 0.15)`。例如高档 0.8 时：

```text
cue + 0.00–0.25 s：0.80
cue + 0.25–0.50 s：0.65
cue + 0.50–0.75 s：0.80
cue + 0.75–1.00 s：0.65
……到 trial 结束停止
```

**每 0.25 秒切换一次档位，完整高低周期为 0.5 秒**。低档为非零时属于强度
起伏，不会完全停转。强度以整数百分比 0–100 写入设备；高档 0 时两档均为 0。

使用主机绝对时钟安排切换，避免把每次 BLE 写入耗时累积到后续切换。浏览器
渲染和 BLE 写入均有延迟，因此这是软件同步，不能保证物理起振与屏幕刷新
毫秒级同时发生。系统记录命令请求与返回时间；GATT 返回成功不代表已测量
马达物理起振。

## 断连与停止

`tactile_MI` 只有在有效 EEG 信号和两个马达均就绪、没有测试运行时，才允许
点击开始。实验正常完成、按 Esc 中断、服务正常关闭时均停止双马达。

正式 trial 期间 EEG/马达断连会终止实验，停止马达，并回退删除当前
未完成 trial 的原始 EEG、事件、震动及反馈记录，只保留完整 trial。
EEG 套接字关闭、连续 3 秒未收到有效 EEG（可配置），或收到不合法的 EEG
数据块均视为信号中断；单纯连上套接字或发送状态消息不算有效信号。
未执行的 trial 仍保留计划行，实际边界重置为 NaN / -1，供之后重做。
准备阶段断连不开始记录（新实验清理临时输出；续采不修改原文件）。
组间休息期间断连会暂停倒计时并自动
重连；所有所需设备就绪、EEG 信号恢复后，操作者点击继续，重新计时 30 秒。

## restart_test：继续原实验

1. 首页选择 `restart_test`。输入原 H5 的绝对路径，或其所在文件夹；文件夹
   自动定位其中的 `motor_imagery.h5`，文件路径则使用指定文件。
2. 点击“解析文件”，显示原模式、受试者、时长、trial 总数、已完成/剩余数量、
   下一个 trial 和休息位置。路径不正确可直接修改后重新解析。解析是只读操作。
3. 点击“继续实验”，进入原模式的设备检查界面。触觉实验仍须连接两侧马达；
   自主学习恢复原控制参数并校验 Teacher 的 SHA256，需要原路径上的同一模型。
4. 调试完成点击“开始采集”，经过准备阶段后才打开原 H5 续写。取消准备或
   返回修改路径不会改动原数据。旧版中断文件中残留的半个 trial 在此时清理。
5. 续采保持原 trial 序列、编号、分组、时长与刺激设置。例如每 20 次休息，
   第 19 次断连保留 18 次：续做 19、20 后休息，再开始 21。若第 20 次之后
   的休息已经完整结束，则直接做 21；未结束的休息重新执行。

EEG 直接追加至原文件，已完成 trial 数据不重写；允许再次断连并多次续采。
结束后 H5 和数据 README 均为 `complete`，最终 H5 不保留被回退尝试的
abort 事件。中断历史仅追加在数据 README 的“中断记录”，记录 trial 号、
保留数量、时间及原因，并在重写 README 时保留。

`actual_time_s`、`wall_time_s` 在保留段之间连续拼接，后者仍包含有效的
组间休息；被删除尝试、离线等待和重新调试时间不计入。采样索引连续不表示
生理信号在断点处物理连续，分析跨断点的信号时应查阅 README 中的记录。
续采不接受不兼容的采样率、通道、显示滤波设置，或不同的真实/演示运行方式。
旧设备的真实 BLE 文件仍可读取，但不允许将新 WebSocket 设备数据续写到其中；
新设备应新建会话。同配置的 WebSocket 文件可按原流程多次续采。
`Esc` 仍保存已有的部分数据；若之后通过 restart_test 继续，未完成 trial
同样先清除并重做。

写零失败时程序尝试主动断开该马达，依赖原 XIAO 固件的“断连停机”行为。
进程被强制杀死、主机断电等情况无法由 Python 保证发出停止命令，实机使用
需保留这一固件行为。

## 数据输出与读取

每次实验输出一个 `motor_imagery.h5` 和一个中文 `README.md`。三种模式的
EEG、事件及 trial 索引结构保持兼容，schema 版本为 **1.4**：

| 路径 | 内容 |
|---|---|
| `/metadata/json` | 受试者信息、模式、震动高档、随机种子、状态和运行环境标记 |
| `/protocol/json` | 时序、滤波、类别、触觉协议及事件编码 |
| `/eeg/raw_uv` | 250 Hz，默认 32 通道，接收原始微伏数据 `[samples, channels]`，float32 |
| `/events/table` | 协议事件时间和对应 EEG 样本索引 |
| `/trials/table` | 随机序列、block_id、cue 等实际采样边界 |
| `/haptics/commands` | 震动命令侧别、百分比、trial_id、请求/返回时间、样本索引及成败 |
| `/self_learning/feedback` | 仅自主学习：窗口边界、模型概率、控制与位置及耗时 |
| `/self_learning/config_json` | 仅自主学习：teacher 哈希、预处理、控制配置及测速结果 |
| `/quality` | 样本覆盖率、时间误差、异常数据包及震动命令失败数 |

准备和设备检查 EEG 不保存。组间休息继续接收、滤波和显示 EEG，但不写入
HDF5，也不补零，之后的数据继续追加到同一个文件。设备检查测试不写入
实验文件；组间测试会记录命令，`trial_id=-1`，`context=test`。

`pure_MI` 的命令表为空。新文件**不创建 `/eeg/filtered_uv`**。
波形仍在内存中做原来的 50 Hz 陷波和两级 5–30 Hz 四阶 Butterworth 带通，
显示范围 ±50 uV，文件中的原始数据不裁剪。
`load_session.py` 同时读取旧文件；新文件的 `filtered_uv` 返回 `None`，
`extract_trial()` 默认取 `raw_uv`。若续采旧模拟文件，保留已有滤波前缀，
此后只追加原始数据；请求未覆盖的滤波 trial 会明确报错。

```bash
python load_session.py /absolute/path/to/motor_imagery.h5
```

```python
from load_session import extract_trial, load_session

session = load_session("/absolute/path/to/motor_imagery.h5")
mode = session["metadata"].get("experiment_mode", "pure_MI")
raw_uv = session["raw_uv"]
trials = session["trials"]
haptic_commands = session["haptic_commands"]
trial_1 = extract_trial(session, 1, window="motor_imagery")
```

分析时使用 `trials["cue_start_sample"]` 定位 cue，采样索引从零开始，切片左闭
右开。命令表的 `sample_index` 对应命令返回时的样本位置；请求/返回墙钟时间
包含休息，可以与 `/events/table` 的 `wall_time_s` 比较。

可选导出事件与 trial 表为 CSV：

```bash
python load_session.py /absolute/path/to/motor_imagery.h5 --export-csv /absolute/path/to/csv_output
```

## 无设备演示与验证

```bash
python app.py --demo
```

三种模式都可演示，EEG 和马达全部模拟，不扫描或控制真实外设。
演示默认 8 通道，`--eeg-channels 32` 可检查全部 32 通道显示。
8 通道演示可直接使用 `--mock-teacher`；32 通道演示需显式设置模型通道映射。
快速检查可以按 20 倍速度执行，建议左右各 1 个 trial：

```bash
python app.py --demo --time-scale 0.05
```

此时时序（含 2 秒震动测试和 0.25 秒切换）按比例缩短；数据仍以协议意义
上的 250 Hz 保存。`/metadata/json` 的 `runtime` 中写入 `demo` 和 `time_scale`。
真实采集禁止时间缩放。

```bash
python -m unittest discover -s tests -v
```

自动验证覆盖原解码/滤波、真实本地 WebSocket 接收、32 通道原值落盘、
单客户端限制、超时/错误数据断连、三种模式、休息、中断回滚及原文件续采。
它不能替代厂家硬件与两侧马达的物理时序验证。

## 目录与硬件参数

```text
app.py                    服务入口、实验状态与 API
config.py                 EEG / 马达 / 时序 / 滤波的集中配置
experiment.py             实验配置、随机序列与协议计划
eeg.py                    原显示滤波、模拟源及保留的旧 BLE 协议工具
ws_eeg.py                 厂家 WebSocket 接收、协议校验、单源及断流检查
haptics.py                双马达连接、测试、定时刺激和停止
storage.py                HDF5 写入及实验 README
restart.py                续采文件校验、完整 trial 前缀和恢复位置
load_session.py           读取、trial 切片及可选 CSV 导出
adfcnn/                   ADFCNN 训练、teacher 模型保存和 Feedback 推理
self_learning/            自主学习原始窗口推理、光标控制与 control.yaml
static/                   无构建依赖的网页界面
templates/                每次实验的中文说明模板
BLE/haptic_control.py     单马达手动调试工具
tests/                    自动验证
requirements.txt          数采与 ADFCNN 的统一 Python 依赖
constraints-tested.txt    本次验证的直接依赖版本
environment.yml           conda 新环境配置
```

更换硬件广播名、UUID 或固定时序时修改 `config.py`。手动检查单侧马达：

```bash
python BLE/haptic_control.py --side left
python BLE/haptic_control.py --side right
```

输入 `0.8` 控制强度，`stop` 停止，`q` 停止并退出。不要同时运行手动调试工具
和需要占用同一马达的采集程序。

端口被占用时可直接换端口：

```bash
python app.py --port 8010
```
