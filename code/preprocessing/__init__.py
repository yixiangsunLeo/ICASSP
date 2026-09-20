"""预处理子包：raw BDF -> [trials, channels, time] 处理后张量。

两种变体（见 docs/PREPROCESSING_SPEC.md）：
  - paper_style：先实现，不做 ICA。
  - eog_ecg_clean：第二阶段，用 ECG/EOG 辅助去伪迹。
现存模块（2026-09-18 剪枝后）：eog_ecg_clean（WBCIC-SHU 2C 正式 npz 的清洗函数，数据出处）、
neuracle_events（bdf 事件解析）、shu_mat（SHU 2022 .mat→npz）、wbci_3c_mat（WBCIC-SHU 3C .mat→npz）。
旧的 pipeline.py / shu_preprocess.py 在 ../_archive/eeg-mi-online_legacy_2026-09-18/code/preprocessing/。
"""
