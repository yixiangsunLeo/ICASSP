---
title: "HANDOFF — 给 3090 / 新机器迁移 AI 的执行手册"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
created: "2026-09-20"
updated: "2026-09-20 20:30"
status: "active"
audience: "the agent on the new 3090 account"
---

# 你是谁、你要做什么

你是 **ICASSP 2027 项目在新机器（3090 账号）上的迁移与训练 Agent**。
用户把你放进这个仓库，是为了让项目在这边 **原样跑起来**，并把实验结果写回 **同一套相对路径**，再经 GitHub 同步到 4090 上的 yuan 树。

**从打开本文件的这一秒起，按下面的 Phase 0 → 13 顺序执行。不要跳步。任何一步 FAIL 就停，把完整终端输出原样发给用户，不要自己改 `code/` 或配方来绕过。**

本文件是你的主程序。细节附录：

| 附录 | 何时读 |
|:---|:---|
| `docs/handoff/00_IDENTITY.md` | 现在立刻读完 |
| `docs/handoff/INSTALL.md` | Phase 4 |
| `docs/handoff/DATA.md` | Phase 5–7 |
| `docs/handoff/TRAIN.md` | Phase 11 |
| `docs/handoff/RESULTS_SYNC.md` | Phase 12 |
| `docs/handoff/PATHS_LOCAL.template.yaml` | Phase 6 |

灵魂记忆（规则，不是步骤）：`AGENTS.md` → `docs/STATUS.md` → `progress.md` 顶部。**先把本文件 Phase 0–2 做完再读灵魂文件，避免还没落地就去改 4090 的作业。**

---

# Phase 0 — 铁律（违反任何一条就停）

1. **不要** `scancel` / `scontrol` 别人的作业，尤其不要碰 4090 集群上 yuan / Linzh 的作业。你只管理 **本账号、本集群** 你自己提交的作业。
2. **不要** 改 `code/`、`scripts/slurm/`、`code/configs/experiments/snn_*.yaml`、`code/configs/experiments/bl_*_zs_lr1e4.yaml` 的训练配方。路径问题用 `paths.local.yaml` + `scripts/repath_manifests.py` 解决，不用改 Python。
3. **不要** 把 `code/configs/paths.local.yaml`、权重 `.pth/.ckpt`、`*.npz`、`*.pt` 推进 GitHub。
4. **不要** 编造结果。没有 `result.json` 的 cell 只能写 planned / running。
5. **不要** 看测试集选模型。论文行规则已锁（见 `AGENTS.md` §4、D-040）。
6. **不要** 换 `run_id`。训练必须写进已经存在的三个目录（见 Phase 11），这样 4090 拉下来路径一字不差。
7. **不要** 为了迁就 3090 显存去改 `batch_size` / `micro_batch_per_model` / `patience` / `monitor`。这些在 `cell_signature` 里，一改就和已完成 cell 无法配对，也会把两种配方写进同一张表。显存不够就 **一卡一进程**，不要改配方。
8. **不要** `git checkout -- .`、`git restore .`、`git stash`、`git clean -fd`、`git push --force`、`git commit --amend`。
9. **不要** 在登录节点训练。分析脚本可以。
10. 仓库必须保持 **private**（含学长未发表的 AdaHGNN 官方文件）。发现变成 public 立刻停并告诉用户。

---

# Phase 1 — 确认你在正确的仓库

```bash
pwd
test -f AGENTS.md && test -f HANDOFF.md && test -d code/models/spiking && echo "TREE OK" || echo "TREE FAIL"
git remote -v
git log -1 --oneline
```

期望：

- 目录里有 `AGENTS.md`、`HANDOFF.md`、`code/models/spiking/`、`code/models/external/`、`.cursor/rules/00-project.mdc`。
- remote 指向 `https://github.com/yixiangsunLeo/ICASSP` 或 SSH 等价地址（`git@github.com:yixiangsunLeo/ICASSP.git` / `git@github.com-yixiangsunLeo:yixiangsunLeo/ICASSP.git`）。
- **不要** clone 旧仓库 `stevencummings341-prog/EEG`（2026-08-28 的树，没有 SNN、没有本文件）。

```bash
git clone -b main git@github.com:yixiangsunLeo/ICASSP.git ICASSP
cd ICASSP
```

clone 完立刻再跑：

```bash
test -d code/models/spiking && test -f scripts/handoff_preflight.sh && echo "SNN TREE OK" || echo "SNN TREE FAIL — 你拿到的是旧 EEG.git，停"
echo "formal result.json: $(find outputs/experiments -path '*/_smoke/*' -prune -o -name result.json -print 2>/dev/null | wc -l)  (expect >= 300 if 4090 already pushed cells)"
echo "bench snapshot result.json: $(find results_snapshot_2026-09-13 -path '*/cells/*/result.json' 2>/dev/null | wc -l)  (expect 587)"
```

`SNN TREE FAIL` 或 `spiking` 目录不存在：停，不要训。
`formal result.json` 为 0：4090 还没把已完成 cell 推进这个仓库。你仍然可以落地环境和数据，但 **不要重跑** STATUS 里已经标齐的条目（会烧光截稿前的卡）。先问用户是否已经 push 了 `result.json`。

然后读：

1. `docs/handoff/00_IDENTITY.md`
2. `AGENTS.md`（5 分钟，规则）
3. `docs/STATUS.md`（现在缺哪些 cell；看「3090 缺口」表）
4. `progress.md` 顶部一条
5. `docs/PLAN_spiking_eegnex_icassp2027.md` §0.5（为什么配方锁死）
6. `docs/PROTOCOL_external_baselines.md`（若你要跑 FM / AdaHGNN）
7. `docs/DECISIONS.md` 里 D-035 / D-034 / D-040 / D-041

---

# Phase 2 — 发现本集群（写下来，后面提交作业要用）

在本机跑，把输出保存到 `logs/slurm/handoff_cluster.txt`（目录不存在就 `mkdir -p logs/slurm`）：

```bash
mkdir -p logs/slurm
{
  echo "==== host ===="; hostname; whoami; date
  echo "==== nvidia ===="; nvidia-smi -L || echo "no nvidia-smi on this node"
  echo "==== slurm ===="; command -v sbatch && sinfo -o '%P %a %G %l %D %T %N' || echo "NO SLURM"
  echo "==== qos ===="; sacctmgr show qos -n format=name,maxtresperuser 2>/dev/null | head
  echo "==== df ===="; df -h "$HOME" . /data /share /mnt 2>/dev/null | head -20
  echo "==== conda ===="; command -v conda; echo "CONDA_ROOT=${CONDA_ROOT:-}"
} | tee logs/slurm/handoff_cluster.txt
```

你必须自己判定并记住：

| 变量 | 怎么定 |
|:---|:---|
| `DATA_ROOT` | 一块 **本机可读可写**、至少空余 **15 G** 的盘。优先项目外（`$HOME/icassp_data` 或 `/data/$USER/icassp`）。不要写进 `/share/workspace2`（这边读不到）。 |
| `PARTITION` | `sinfo` 里能提交 GPU 作业的分区名。**不要**沿用 `gpu2node`/`gpu3node`，那是 4090 集群的。 |
| `CONDA_ROOT` | 本机 conda 安装前缀。没有就 Phase 4 自己装 miniconda 到 `~/miniconda3`。 |
| 有没有 Slurm | 没有 Slurm：GPU 交互节点上直接 `python code/run.py ...`，仍然禁止登录节点训练。 |

把这四项写进 **会被 git 跟踪** 的 `docs/handoff/CLUSTER_3090.md`（从模板复制），不要只写 `logs/`（logs 不进 git，4090 拉不到）：

```bash
cp docs/handoff/CLUSTER_3090.template.md docs/handoff/CLUSTER_3090.md
# 填 PARTITION / DATA_ROOT / CONDA_ROOT / 账号 / 是否有 Slurm
```

并 `export DATA_ROOT='<绝对路径>'` 写进本账号的 `~/.bashrc`。后面每个新 shell 都要有它。

---

# Phase 3 — 向用户要数据（你读不到 4090 的 workspace）

processed 数据 **不在 GitHub**（约 9.6 G）。你必须向用户给出下面这段，让他 **在 4090 / yuan 账号** 上执行，推到你的 `DATA_ROOT`。你自己不要假装数据已经在。

把下面三个占位符换成真实值，**整段**发给用户（源路径末尾斜杠必须保留，否则会多嵌一层目录）：

```bash
# 在 4090 login01、yuan 账号上跑。
HOST='<3090_USER@HOST>'          # 例：you@3090-login
DATA_ROOT='<DATA_ROOT>'          # 3090 上的绝对路径，例：/data/you/icassp_data
REPO='<REPO>'                    # 3090 上本仓库的绝对路径，例：/home/you/ICASSP

rsync -aH --info=progress2 \
  /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/ \
  "${HOST}:${DATA_ROOT}/WBCIC_SHU/processed/eog_ecg_clean/"
rsync -aH --info=progress2 \
  /share/workspace2/moto_imagination/SHU/processed/npz_clean/ \
  "${HOST}:${DATA_ROOT}/SHU/processed/npz_clean/"
rsync -aH --info=progress2 \
  /share/home/yuan/SYX/ICASSP/outputs/processed/wbci_shu_3c_mat_clean/ \
  "${HOST}:${REPO}/outputs/processed/wbci_shu_3c_mat_clean/"
# 若还要跑 LaBraM / EEGPT / CBraMod：
mkdir -p weights   # 这句话在 3090 上先跑
rsync -a \
  /share/home/yuan/SYX/Benchmark/weights/labram-base.pth \
  /share/home/yuan/SYX/Benchmark/weights/eegpt.ckpt \
  /share/home/yuan/SYX/Benchmark/weights/CBraMod.pth \
  "${HOST}:${REPO}/weights/"
```

3C 必须落在仓库内 `outputs/processed/wbci_shu_3c_mat_clean/`，和 4090 相对位置一致。

**数据没到齐之前不要安装完就开训。** 继续 Phase 4（环境可以并行装），Phase 5 再确认数据。

期望体积（`du -sh`）：

| 树 | 大约 |
|:---|:---|
| `$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean` | 6.2 G |
| `$DATA_ROOT/SHU/processed/npz_clean` | 1.4 G |
| `$REPO/outputs/processed/wbci_shu_3c_mat_clean` | 2.0 G |
| `$REPO/weights/{labram-base.pth,eegpt.ckpt,CBraMod.pth}` | 93 M + 389 M + 19 M |

---

# Phase 4 — 安装环境

完整命令见 `docs/handoff/INSTALL.md`。这里是你必须跑通的最短序列。

```bash
# 若没有 conda：先装 miniconda 到 ~/miniconda3，然后：
# source ~/miniconda3/etc/profile.d/conda.sh

conda create -n mi_torch_cu118 python=3.10.18 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch_cu118

# 永远从 PyTorch 官方 index 装 GPU 轮，禁止 conda install pytorch（会装成 CPU）。
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

pip install \
  numpy==2.2.5 scipy==1.15.3 scikit-learn==1.7.1 pandas==2.2.3 \
  matplotlib==3.10.0 PyYAML==6.0.2 tqdm==4.67.1 pillow==11.3.0 \
  einops==0.8.1 timm==1.0.19 dhg==0.9.7 pytest==9.1.1

python - <<'PY'
import torch, timm, einops, torchaudio
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert str(torch.__version__).startswith("2.7.1")
PY
```

3090 是 sm_86，cu118 轮可用。驱动过旧再考虑 cu121，**不要把 torch 降到 2.4 以下**。登录节点 `cuda_avail=False` 可以，GPU 作业里必须是 True。

作业脚本通过 `CONDA_ROOT` 找 conda，不要改 `scripts/slurm/_common.sh`：

```bash
export CONDA_ROOT="$(conda info --base)"
export MI_ENV=mi_torch_cu118
```

把这两行写进 `~/.bashrc` 或你提交作业用的环境。

---

# Phase 5 — 确认数据已落地

```bash
: "${DATA_ROOT:?export DATA_ROOT to an absolute path first}"
echo "DATA_ROOT=$DATA_ROOT"
du -sh \
  "$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean" \
  "$DATA_ROOT/SHU/processed/npz_clean" \
  outputs/processed/wbci_shu_3c_mat_clean
ls "$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv"
ls "$DATA_ROOT/SHU/processed/npz_clean/processed_manifest.csv"
ls outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
```

任一 FAIL：回到 Phase 3，向用户要数据。不要自己生成 2C/SHU（那是 BDF+ICA，天级）。3C 若只有官方 `.mat`、没有 processed 树，才允许跑 `python scripts/preprocess_wbci_3c.py --mat-root <mat> --out-root outputs/processed/wbci_shu_3c_mat_clean`（会写出本机绝对路径，可跳过 3C 的 sed）。

---

# Phase 6 — 写 `paths.local.yaml`（每台机器一份，不进 git）

```bash
cp docs/handoff/PATHS_LOCAL.template.yaml code/configs/paths.local.yaml
# 把文件里所有 <DATA_ROOT> 换成 Phase 2 的真实绝对路径
```

用编辑器或：

```bash
# 例子：DATA_ROOT=/data/icassp
sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" code/configs/paths.local.yaml
grep CHANGE code/configs/paths.local.yaml && echo "STILL PLACEHOLDER" || echo "paths filled"
grep -E '/share/workspace2|/share/home/yuan' code/configs/paths.local.yaml && echo "STILL OLD CLUSTER"
```

`wbci_3c_processed_manifest` 保持相对路径 `outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv`。
`pretrained_weights_dir: "weights"` 保持相对。

**不要** `git add` 这个文件。

---

# Phase 7 — 改写 manifest 绝对路径（不做这一步，训练会去读 4090 的盘）

代码 **不会** 重定根。csv 里的 `npz_path` 是唯一数据地址。拷完必须改前缀。

```bash
: "${DATA_ROOT:?export DATA_ROOT to an absolute path first}"
export REPO="$(pwd)"

# 先 dry-run，确认将改写的条数 > 0，再去掉 --dry-run 重跑同一条。
python scripts/repath_manifests.py \
  --csv "$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv" \
  --rewrite "/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean=${DATA_ROOT}/WBCIC_SHU/processed/eog_ecg_clean" \
  --expect-ok 148 --expect-subjects 51 --expect-rows 153 --dry-run --verify

python scripts/repath_manifests.py \
  --csv "$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv" \
  --rewrite "/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean=${DATA_ROOT}/WBCIC_SHU/processed/eog_ecg_clean" \
  --expect-ok 148 --expect-subjects 51 --expect-rows 153 --verify

python scripts/repath_manifests.py \
  --csv "$DATA_ROOT/SHU/processed/npz_clean/processed_manifest.csv" \
  --rewrite "/share/workspace2/moto_imagination/SHU/processed/npz_clean=${DATA_ROOT}/SHU/processed/npz_clean" \
  --expect-ok 125 --expect-subjects 25 --expect-rows 125 --verify

# 3C：旧前缀是 yuan 家目录。新前缀 = 本仓库绝对路径下的同一相对目录。
python scripts/repath_manifests.py \
  --csv "${REPO}/outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv" \
  --rewrite "/share/home/yuan/SYX/ICASSP/outputs/processed/wbci_shu_3c_mat_clean=${REPO}/outputs/processed/wbci_shu_3c_mat_clean" \
  --expect-ok 33 --expect-subjects 11 --expect-rows 33 --verify
```

三条都必须打印 `OK`。若 `nothing matched` 但 verify 仍看到 `/share/workspace2`：OLD 前缀写错了，对照 csv 第一行 `npz_path` 再改，不要手改几百行。

**禁止** 删掉 status=`failed` 的 2C 五行，也 **禁止** 改 `subject_id`。少一个被试，`make_subject_folds` 会重排所有折，和 18 个 ANN、已完成 cell 全部对不上。

---

# Phase 8 — 权重（只跑 FM 才必须）

```bash
mkdir -p weights
ls -lh weights/labram-base.pth weights/eegpt.ckpt weights/CBraMod.pth
```

缺文件且要跑 `labram` / `eegpt` / `cbramod`：向用户再要 Phase 3 最后那段 rsync。
只跑 EEGNeX / SNN / EEGNet / ATCNet / FBCNet / EEG-Deformer / TCFormer / AdaHGNN-MI：可以没有这三项。

sha256（拷完应对上，见 `code/models/external/README.md`）：

| 文件 | sha256 |
|:---|:---|
| `labram-base.pth` | `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` |
| `CBraMod.pth` | `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` |
| `eegpt.ckpt` | `f406ec002fd56f618f323c170b9c9e68997ea3299b33575611cb3e26fb1be273` |

`weights/` 在 4090 上是指向 `../Benchmark/weights` 的软链。这边必须是 **真目录 + 真文件**，不要做那个软链。

---

# Phase 9 — 预检（不过就不能训）

```bash
chmod +x scripts/handoff_preflight.sh scripts/sync_results_git.sh
export CONDA_ROOT="$(conda info --base)"
export MI_ENV=mi_torch_cu118
conda activate mi_torch_cu118
bash scripts/handoff_preflight.sh
# 若 3C 数据还在路上、你只想先训 2C：PREFLIGHT_COHORTS=2c,shu bash scripts/handoff_preflight.sh
```

必须看到最后一行 `PREFLIGHT OK`。FAIL 就停。只对已经 `OK` 的队列开训。

可选（CPU，可在登录节点）：

```bash
pytest tests/models/test_spiking.py tests/models/test_external_integration.py tests/models/test_adahgnn.py -q
```

---

# Phase 10 — 冒烟（1 折 2 epoch，写到 `_smoke/`，不进正式 run）

在 **GPU 节点** 上（或 `srun` 进 GPU）：

```bash
conda activate mi_torch_cu118
python code/run.py \
  --config code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml \
  --models eegnex \
  --split-seeds 41 \
  --folds-subset 0 \
  --tag-suffix smoke3090 \
  --max-epochs 2 \
  --no-resume \
  --out outputs/experiments/_smoke/smoke3090 \
  --ckpt-dir checkpoints/_smoke/smoke3090
```

`--ckpt-dir` 和 `--no-resume` **不能省**：否则 2 epoch 的权重会写进正式 `checkpoints/wbci_shu/snn_wbci2c_zs_lr1e4_v1/eegnex__split41__fold0__seed0/`，正式 cell 会因 `cell_signature` 里的 `max_epochs` 对不上而拒绝续跑。

成功：`outputs/experiments/_smoke/smoke3090/` 下出现 cell，且对应 `failures__*.json` 的 `n_cells_failed` 为 0。
失败：看该目录和终端最后 80 行，**不要**改配方。把报错发用户。

`_smoke/` 被 gitignore，不要提交。

---

# Phase 11 — 正式训练（默认：先落地，等用户点名再跑）

**不要**把 4090 / Linzh 正在跑或已跑的作业自动搬过来补齐。那些跑不跑 3090，**只有用户一句话才能开**。

用户后续主线（点名之后才跑）：**S-EEGNeX-D**（`seegnex@hplif`）和 **AdaHGNN-MI**（`adahgnn@mi` / `adahgnn@mi_mix`）。学长改完的代码会先在 4090 这棵树整合，再 `git pull` 到你这边跑。没点名之前：Phase 4–9 把环境、数据、路径做完，**停在 `PREFLIGHT OK`，不要 sbatch。**

三个正式目录，**一个字都不要改**：

```
outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1
outputs/experiments/shu/snn_shu_zs_lr1e4_v1
outputs/experiments/wbci_shu/snn_wbci3c_zs_lr1e4_v1
checkpoints/wbci_shu/snn_wbci2c_zs_lr1e4_v1
checkpoints/shu/snn_shu_zs_lr1e4_v1
checkpoints/wbci_shu/snn_wbci3c_zs_lr1e4_v1
```

对应配置已经写死 `run_id` / `output_dir` / `checkpoint_dir`。**不要加 `--out`。** 已有 `result.json` 且 `cell_signature` 一致的 cell 会自动跳过。

先数盘上有什么：

```bash
for r in wbci_shu/snn_wbci2c_zs_lr1e4_v1 shu/snn_shu_zs_lr1e4_v1 wbci_shu/snn_wbci3c_zs_lr1e4_v1; do
  printf '%-45s %s\n' "$r" "$(find outputs/experiments/$r/cells -name result.json 2>/dev/null | wc -l)"
  ls outputs/experiments/$r/cells 2>/dev/null | sed 's/__.*//' | sort | uniq -c
done
```

用户点名之后再读 `docs/STATUS.md`，只跑他写下的模型。没点名就停。

点名之后有 Slurm 时（`PARTITION` 换成 Phase 2 的真名）。例如用户说跑 S-EEGNeX-D：

```bash
export PARTITION='<本集群 GPU 分区>'
export CONDA_ROOT="$(conda info --base)"
export MI_ENV=mi_torch_cu118
export N_LINKS=2
DRY_RUN=1 bash scripts/slurm/submit_snn.sh \
  code/configs/experiments/snn_wbci2c_zs_lr1e4.yaml c2d "seegnex@hplif"
```

用户说跑 AdaHGNN 时才用 `scripts/slurm/baselines_plan_adahgnn.txt`。没有 Slurm：GPU 节点上 `python code/run.py --config ... --models ...`，不要 `--out`。

作业墙钟不是 48 h 时：用环境变量 / `sbatch -t` 覆盖，**不要改** `scripts/slurm/*.sbatch` 里写死的 4090 默认值（那些文件同时被 4090 作业使用的副本若被 push 回去会害 yuan）。本机需要不同默认时，用命令行 `sbatch -p ... -t ...` 包一层，或设 `PARTITION`。

盯进度：

```bash
find outputs/experiments/wbci_shu/snn_wbci2c_zs_lr1e4_v1/cells -name result.json | wc -l
# 失败看 run 目录下 failures__<tag>.json 的 n_cells_failed，不要翻 slurm 日志猜
```

---

# Phase 12 — 把结果推回 GitHub（4090 才能读到）

训练仍写在上面那些相对路径。Git 只跟踪小文件：`result.json`、`failures__*.json`、`resolved_*`、`outputs/analysis/**/*.csv|md`。`.pt` / `.npz` **永远不进 git**。

每批 cell 完成后：

```bash
# 先更新 STATUS / progress（Phase 13），再：
SYNC_COMMIT=1 SYNC_PUSH=1 bash scripts/sync_results_git.sh
```

脚本会拒绝：二进制、`paths.local.yaml`、force-push。
remote 必须是 `yixiangsunLeo/ICASSP`。你这边 clone 出来的话 `origin` 就是它。

4090 上 yuan 会 `git pull` 这个 remote，读到 **同一相对路径** 下的 `result.json`。

---

# Phase 13 — 活文档（每次实质动作）

- `docs/STATUS.md`：3090 作业、cell 数、下一步。加上「3090 集群」：分区、`DATA_ROOT`、账号。
- `progress.md`：顶部追加一条（日期时刻、做了什么、下一步）。
- `docs/DECISIONS.md`：只有改了数字口径才加 D-xxx。迁机本身已经是 D-041，你不要再为「换了机器」新开一条。
- **不要** 往 `AGENTS.md` 堆状态数字。

---

# 路径对照（出问题先看这里）

| 东西 | 4090 上 | 你这边必须变成 |
|:---|:---|:---|
| 2C npz | `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/...` | `$DATA_ROOT/WBCIC_SHU/processed/eog_ecg_clean/...` |
| SHU npz | `/share/workspace2/moto_imagination/SHU/processed/npz_clean/...` | `$DATA_ROOT/SHU/processed/npz_clean/...` |
| 3C npz | `/share/home/yuan/SYX/ICASSP/outputs/processed/wbci_shu_3c_mat_clean/...` | `$REPO/outputs/processed/wbci_shu_3c_mat_clean/...` |
| 实验 YAML | 逻辑键 `processed_manifest` 等 | **不改 YAML**，改 `paths.local.yaml` |
| 结果 | `outputs/experiments/{wbci_shu,shu}/snn_*_zs_lr1e4_v1/cells/` | **同一相对路径** |
| 权重 | `weights -> ../Benchmark/weights` | 真目录 `weights/` |
| conda | `/share/home/yuan/.conda/envs/mi_torch_cu118`（700，你读不到） | 本机重建同名环境 |
| Slurm 分区 | `gpu2node` / `gpu3node` | 本集群 `sinfo` 里的名字，用 `PARTITION=` |

---

# 你不要做的事（再列一遍）

见 `docs/handoff/00_IDENTITY.md`。补充：

- 不要把 2C/SHU/3C 的 npz 塞进 git。
- 不要为了「跑通」把 numpy 降到 1.x。
- 不要 `conda install pytorch`。
- 不要在没 `PREFLIGHT OK` 时提交全量作业。
- 不要把这边的 `paths.local.yaml` 拷回 4090，也不要从 4090 把那份 `paths.local.yaml` 拷过来。
