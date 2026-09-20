---
title: "handoff — 环境安装（你来执行）"
created: "2026-09-20"
status: "active"
audience: "3090 migration AI"
---

# 环境：重建，不要拷

4090 上的 `/share/home/yuan/.conda/envs/mi_torch_cu118` 在家目录 `700` 里，你读不到；conda 环境也不可搬。按下面重建。名字必须仍是 `mi_torch_cu118`（`scripts/slurm/_common.sh` 默认 `MI_ENV`）。

# 1. conda

没有 conda：

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
```

已有 conda：`export CONDA_ROOT="$(conda info --base)"`。

# 2. 建环境 + GPU torch

```bash
conda create -n mi_torch_cu118 python=3.10.18 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch_cu118
```

**禁止** `conda install pytorch`（会装成 CPU，且旧环境曾经发生过 conda CPU 2.6.0 被 pip 轮盖住的脏状态，不要复现）。

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118
```

驱动 `< 450`：停，让用户升驱动。不要降 torch。代码用了 `torch.amp.GradScaler("cuda")` 和显式 `weights_only=`，需要 torch ≥ 2.4。

# 3. 运行库（版本已在 4090 的 `mi_torch_cu118` 核实）

```bash
pip install \
  numpy==2.2.5 scipy==1.15.3 scikit-learn==1.7.1 pandas==2.2.3 \
  matplotlib==3.10.0 PyYAML==6.0.2 tqdm==4.67.1 pillow==11.3.0 \
  einops==0.8.1 timm==1.0.19 dhg==0.9.7 pytest==9.1.1
```

| 包 | 谁要用 |
|:---|:---|
| `einops` | TCFormer、LaBraM、EEG-Deformer |
| `timm==1.0.19` | LaBraM 官方文件 import `timm.models.layers` |
| `torchaudio` | LaBraM / EEGPT / CBraMod 重采样 |
| `dhg==0.9.7` | 注册表里的原版 `adahgnn`（不跑）。`adahgnn@mi` **不 import dhg**，但装上以免误跑原版崩 |
| `mne` | **只有重做预处理才需要**，训练 processed npz 不需要 |

不要装：`pytorch_lightning`、`spikingjelly`、`braindecode`、`transformers`（TCFormer 的 Lightning 文件未被 import；SNN 自包含）。

# 4. 验收

```bash
python - <<'PY'
import torch, torchvision, torchaudio, numpy, timm, einops
print("torch", torch.__version__, "cuda_build", torch.version.cuda, "avail", torch.cuda.is_available())
assert str(torch.__version__).startswith("2.7.1")
assert numpy.__version__.startswith("2.")
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

登录节点 `avail False` 可以。GPU 作业里必须 True。`scripts/slurm/_common.sh` 的 `mi_cuda_failfast` 在 CPU 上会 `exit 2`，这是故意的。

# 5. 作业如何找到这个环境

不要改 `_common.sh`。提交前：

```bash
export CONDA_ROOT="$(conda info --base)"
export MI_ENV=mi_torch_cu118
```

脚本查找顺序：`$CONDA_ROOT` → `/share/software/anaconda3/2024.10`（4090 路径，你这边没有）→ `~/miniconda3` → `~/anaconda3` → `PATH` 上的 conda。

# 6. 常见翻车

- matplotlib 家目录不可写：`export MPLCONFIGDIR="$PWD/.cache/mpl"`。
- 必须在 **仓库根** 跑 `python code/run.py`。包名 `code` 会和标准库撞；`pyproject.toml` 不安装包，靠 `pythonpath = ["."]`。
- numpy 必须 2.x。不要为某个旧包把 numpy 降回去。
