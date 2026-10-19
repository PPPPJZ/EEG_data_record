"""生成自主学习会话的中文数据说明。"""
from config import SELF_LEARNING_REST_SECONDS, SELF_LEARNING_TASK_SECONDS, SAMPLE_RATE_HZ


def write_readme(config, *, status, quality=None, reason=""):
    quality = quality or {}
    text = f"""# 自主学习 EEG 实验记录

受试者：{config.subject_id}；日期：{config.experiment_date}；模式：self_learning。
描述：{config.description or '未填写'}。
状态：{status}；终止原因：{reason or '无'}。
学习 trial：{config.total_trials}；每 {config.break_every_trials} 个 trial 组间休息。
Teacher：{config.teacher_model_path}。

开始采集后先准备 5 秒，此时不保存 EEG。每个 trial 共 10 秒：
0–{SELF_LEARNING_REST_SECONDS:g} 秒休息并自主决定想象哪一侧；
随后 {SELF_LEARNING_TASK_SECONDS:g} 秒运动想象，显示蓝色小球，不显示左右指令。
每次控制开始时重置小球、EMA 和原始缓冲；收齐本次想象 750 点后开始推理，
即通常在 trial 第 6 秒出现首次反馈，之后每 40 ms 尝试推理最近的完整窗口。
实际间隔及控制参数以 H5 self_learning/config_json 为准。

受试者的真实选择没有记录。trials/table 的 class_id=-1 表示未知，
teacher 概率和预测类别只表示模型判断，不能用于准确率评估或直接当作训练标签。
模型在本阶段保持固定，不进行再训练或权重更新。

组间休息：3 秒闭眼提示，再监控波形 30 秒；剩余 6 秒提示音。
暂停后继续重新计时 30 秒，断连要求信号恢复并手动继续。
组间休息不写 EEG。Esc 中止并保留已采集数据。

H5 为唯一机器可读数据源：

| 路径 | 内容 |
|---|---|
| eeg/raw_uv | 250 Hz、CH1–CH8、微伏，[samples,8] 原始数据 |
| eeg/filtered_uv | 50 Hz 陷波和两级 5–30 Hz 带通，仅用于波形显示 |
| trials/table | class_id=-1；cue_start_sample 为运动想象控制起点，无 cue，因此 cue_end_sample 相同 |
| events/table | 230=self_learning_control_start；其他休息/开始/结束事件沿用原协议 |
| self_learning/config_json | 控制参数、teacher 路径/哈希、推理设备、预处理、模型测试和模拟标志 |
| self_learning/feedback | 窗口起止样本、模型类别、P(Right)、EMA、速度、位置、耗时及时间 |
| metadata/json、protocol/json、quality | 信息、协议和质量 |

模型只使用 raw_uv：先取 750 点窗口，再按 teacher 参数做窗口内 40 Hz FIR
低通和逐通道标准化。不会使用 filtered_uv，也不会跨 trial 或组间休息拼窗口。
window_end_sample 为左闭右开切片的右边界。反馈耗时包含预处理和分类，
不包含采样积累、加载和浏览器渲染；两条时间轴均为秒，wall_time_s 包含休息。
光标坐标归一化，左侧 0、右侧 1；速度单位为屏幕宽度/秒。

计划采样点：{int(config.expected_duration_seconds * SAMPLE_RATE_HZ)}。
实际采样点：{quality.get('recorded_sample_count', '未结束')}；
反馈数：{quality.get('feedback_prediction_count', '未结束')}；
完成 trial：{quality.get('completed_trials', '未结束')}。

读取：from load_session import load_session
"""
    path = config.output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path
