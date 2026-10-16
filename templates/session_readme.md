# 运动想象 EEG 实验记录

## 实验信息

| 字段 | 内容 |
|---|---|
| 受试者编号 | {subject_id} |
| 实验日期 | {experiment_date} |
| 实验描述 | {description} |
| 实验模式 | `{experiment_mode}` |
| 数据状态 | `{status}` |
| 中止原因 | {termination_reason} |
| 左手 trial | {left_trials} |
| 右手 trial | {right_trials} |
| trial 总数 | {total_trials} |
| 每 N 个 trial 休息 | {break_every_trials} |
| 计划组间休息次数 | {scheduled_breaks} |
| 随机种子 | `{random_seed}` |

随机序列中 `L` 表示左手运动想象，`R` 表示右手运动想象：

```text
{sequence}
```

## 实验协议

{haptic_protocol}

设备检查阶段只显示实时滤波波形，不保存 EEG。点击“开始采集”后先进入
5 秒准备阶段，显示“准备采集”、白色十字架和倒计时；此阶段不写入 EEG。

正式记录从第一个 trial 的休息阶段开始。每个 trial 固定为 8 秒：

| trial 内时间 | 阶段 | 屏幕内容 | 是否记录 |
|---:|---|---|---|
| 0.0–2.5 s | 休息 | “休息”、剩余 trial 数、2.5 秒进度条 | 是 |
| 2.5–4.0 s | cue | 左/右白色箭头、运动想象进度条 | 是 |
| 4.0–8.0 s | 运动想象 | 黑屏，仅保留运动想象进度条 | 是 |

运动想象窗口定义为 `[cue_start_sample:trial_end_sample]`，总长 5.5 秒。
箭头显示窗口为 `[cue_start_sample:cue_end_sample]`，后续纯黑运动想象窗口为
`[cue_end_sample:trial_end_sample]`。所有采样切片均为 Python 风格左闭右开区间。

每完成 {break_every_trials} 个 trial 且后面仍有 trial 时，进入一次组间休息：

1. 黑屏显示“请闭眼休息30s”，持续 3 秒。
2. 随后进入实时波形监控页，开始 30 秒倒计时。
3. 剩余 6 秒时播放提示音，倒计时归零后开始下一个 trial。
4. 暂停只冻结倒计时，EEG 仍持续接收、滤波和显示；点击继续会重新开始
   完整的 30 秒倒计时。
5. 组间休息期间的 EEG **不写入 HDF5，也不补零**。休息前后的 block 在
   EEG 数组中紧邻，必须通过 `block_id` 或休息事件识别时间断点。

## 信号与滤波

- 设备：`Motren_E`
- 通道：`CH1`–`CH8`
- 采样率：250 Hz
- 物理单位：uV
- `/eeg/raw_uv`：软件滤波前的原始微伏数据
- `/eeg/filtered_uv`：50 Hz 陷波后，串联两次 5–30 Hz 四阶 Butterworth 带通
- 实时显示范围：±50 uV，超出部分只在显示时裁剪，文件内数据不裁剪

## HDF5 数据结构

`motor_imagery.h5` 是本次实验唯一的机器可读权威数据源。

| 路径 | 内容 |
|---|---|
| `/metadata/json` | 受试者信息、随机种子、实验状态和最终质量信息 |
| `/protocol/json` | 时序、采样率、通道、滤波和事件编码 |
| `/eeg/raw_uv` | `[samples, 8]` 原始 EEG，float32 |
| `/eeg/filtered_uv` | `[samples, 8]` 滤波 EEG，float32 |
| `/events/table` | 实际发生的事件、计划时间、实际时间及采样点 |
| `/trials/table` | 完整随机序列、`block_id` 及每个 trial 的直接切片边界 |
| `/haptics/commands` | 震动命令、返回时间和样本索引；pure_MI 中为空 |
| `/quality` | 样本覆盖率、时间偏差、异常包和终止信息 |

类别编码：

| `class_id` | 类别 |
|---:|---|
| 0 | left_hand |
| 1 | right_hand |
| -1 | 不适用 |

事件编码：

| `event_code` | 事件 |
|---:|---|
| 100 | recording_start |
| 110 | trial_start_rest |
| 210 | cue_left |
| 220 | cue_right |
| 300 | cue_end_imagery_black_start |
| 400 | trial_end |
| 500 | break_instruction_start |
| 510 | break_monitor_start |
| 520 | break_paused |
| 530 | break_countdown_restarted |
| 540 | break_chime |
| 550 | break_end |
| 560 | break_device_disconnected |
| 570 | break_signal_restored |
| 900 | recording_complete |
| 990 | recording_aborted |

`events/table` 的 `planned_time_s` 是不包含组间休息的协议采集时间；
`actual_time_s` 是服务器累计的有效采集时间，也不包含组间休息；
`wall_time_s` 是包含组间休息和人工暂停的真实经过时间。`sample_index` 是
事件发生时已写入的采样点位置。休息相关事件共享同一采样边界，能够证明该
期间没有 EEG 被写入。

`/haptics/commands` 的 `motor_id` 为 0（HapticL）或 1（HapticR），
`percentage` 是实际发送的 0–100 整数。`context` 区分 `trial`、`test`、
`connection` 和 `stop`；测试及连接命令的 `trial_id` 为 -1。
`requested_wall_time_s` 是开始 GATT 写入的时间，`acknowledged_wall_time_s`
是写入返回或失败的时间，两者均相对于正式记录起点、包含组间休息。
`sample_index` 是写入返回时收到的累计 EEG 样本位置。
`success` 为 false 时，`error` 记录错误，包括取消。

时序采用主机软件调度，浏览器显示和 BLE 写入均存在延迟，GATT 成功返回
不等同于马达物理起振。设备检查阶段的测试不写入实验文件；组间测试会记录
命令，但该阶段依然不保存 EEG。HDF5 `/metadata/json` 的 `runtime` 标识
模拟模式和时间缩放，模拟命令不代表真实硬件刺激。

分析时应直接使用 `/trials/table` 中的 `cue_start_sample`，无需根据时间反推：

```python
cue_index = int(trial["cue_start_sample"])
segment = filtered_uv[cue_index:cue_index + int(t_seconds * 250)]
```

## 质量汇总

| 指标 | 数值 |
|---|---:|
| 计划记录时长 | {expected_duration_s} s |
| 计划采样点/通道 | {expected_samples} |
| 实际有效采集时长 | {actual_duration_s} s |
| 实际墙钟时长（包含休息） | {wall_duration_s} s |
| 实际采样点/通道 | {recorded_samples} |
| 数据覆盖率 | {coverage} |
| 采样计数与实际时间偏差 | {sample_time_error_ms} ms |
| 正式记录期间异常数据包 | {malformed_packets} |

## 读取

在项目目录中运行：

```bash
python load_session.py /absolute/path/to/motor_imagery.h5
```

Python 中可以直接读取：

```python
from load_session import load_session

session = load_session("/absolute/path/to/motor_imagery.h5")
raw_uv = session["raw_uv"]
filtered_uv = session["filtered_uv"]
events = session["events"]
trials = session["trials"]
metadata = session["metadata"]
haptic_commands = session["haptic_commands"]
```
