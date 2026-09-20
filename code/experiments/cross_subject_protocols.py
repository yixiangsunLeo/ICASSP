"""Cross-subject (subject-independent) protocol for the end-to-end foundation models.

This is the new mainline track (2026-08-04): train the 5 S4/DINO-DualCD variants
end-to-end on WBCIC-SHU and SHU **separately** (never merged — 58ch vs 32ch, and each
dataset keeps its own baseline), and evaluate on **held-out subjects**.

Split policy (all subject-level; a subject never appears on two sides)
----------------------------------------------------------------------
``protocol``:
  * ``loso``           — leave-one-subject-out: one fold per subject, test = that subject.
  * ``kfold_subject``  — subjects shuffled by ``split_seed`` then cut into ``n_folds``
                         groups; each group is the test set once. Cheaper than LOSO.
  * ``holdout``        — a single fold: explicit ``test_subjects`` (and optionally
                         ``val_subjects``) from the config, or a ``test_fraction`` cut.

``val_mode``:
  * ``subjects`` (default) — the validation set is a few *held-out training subjects*.
    Model selection then never sees a trial from a subject it will be scored on, and it
    also never sees the test subject. This is the strict setting.
  * ``trials``            — a stratified trial slice of the training subjects. Weaker
    (the same subjects appear in train and val) but common in the literature; kept
    available so we can match whatever protocol the advisor picks.
  * ``sessions``          — paper-aligned: same non-test subjects for train and val, but
    different sessions (e.g. train ``ses-01``+``ses-02``, val ``ses-03``; test subject
    keeps all sessions). Matches the SHUv5 / WBCIC-SHU 3C LOSO write-up.

Leakage rules enforced here
---------------------------
  * test subjects are excluded from both train and val, asserted per fold;
  * under ``val_mode=sessions``, train/val may share subjects but session sets are disjoint;
  * the test subject's labels are used **only** for the final metric computation;
  * normalization is per-trial (fit-free), so no statistic ever crosses the split —
    see ``code.models.eeg_foundation.adapter.normalize_trials``;
  * every fold's subject lists are written to ``splits/`` as JSON.

Per cell = (model, fold, seed). Each cell writes exactly two checkpoints
(``best.pt`` / ``last.pt``) plus a ``result.json`` completion marker, so an interrupted
sweep resumes: finished cells are skipped, a partially trained cell continues from its
``last.pt``. Both the best and the last checkpoint are scored on the test subjects, so
"best-epoch" and "final-epoch" numbers are reported side by side.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from ..datasets.session_splits import SessionRecord, group_by_subject, normalize_labels
from ..models.eeg_foundation.adapter import normalize_trials
from ..models.model_specs import safe_tag
from ..models.registry import build_model
from ..training.e2e_trainer import BEST_CKPT, LAST_CKPT, CellPaths, E2ESpec, train_end_to_end
from ..training.trainer import predict
from ..utils.io import load_session_npz, save_json
from ..utils.paths import ses_id
from ..utils.seed import set_seed
from .session_protocols import evaluate_predictions

PROTOCOLS = ("loso", "kfold_subject", "holdout")
VAL_MODES = ("subjects", "trials", "sessions")

#: Bump whenever ``cell_signature``'s payload changes. A stored cell written under an older
#: version cannot be compared against a signature computed by this code, so a resume attempt
#: must fail loudly rather than silently mix two recipes into one table.
#:
#: v2 (2026-08-27) added the early-stopping settings — see ``_stopping_identity``. Under v1
#: a run that lowered ``patience`` silently reused cells trained with the old value, putting
#: two different model-selection rules in one results table with no warning.
SIGNATURE_VERSION = 2


def _stopping_identity(spec: E2ESpec) -> Dict[str, object]:
    """The parts of the stopping rule that define a cell's identity.

    ``early_stopping_patience`` **always** counts: it decides which epoch becomes
    ``best.pt``, so two cells with different patience can report different results from the
    very same optimization trajectory. That is the silent failure v2 exists to prevent.

    ``max_epochs`` counts **only when a scheduler is active**, because its role depends on
    the scheduler:

    * ``scheduler: none`` (this project's recipe) — it is a pure upper bound. Raising it and
      resuming is a strict continuation of the same trajectory, which is a legitimate and
      useful operation (and how a wall-clock-killed job is restarted). Including it would
      forbid that for no methodological gain.
    * ``scheduler: cosine`` — it *is* the annealing horizon, so changing it changes the
      learning rate at every remaining step. A cell resumed under a different horizon is on
      a different trajectory and must not be mixed with the original.
    """
    ident: Dict[str, object] = {"early_stopping_patience": spec.early_stopping_patience}
    if str(spec.scheduler or "none").lower() not in ("none", ""):
        ident["max_epochs"] = spec.max_epochs
    return ident
# Paper-aligned default for WBCIC-SHU / SHUv5 3C LOSO (train ses1+2, val ses3).
DEFAULT_TRAIN_SESSIONS = ("ses-01", "ses-02")
DEFAULT_VAL_SESSIONS = ("ses-03",)

CROSS_SUBJECT_RESULT_COLUMNS: List[str] = [
    "dataset", "model", "protocol", "split_seed", "fold", "seed",
    "n_train_subjects", "n_val_subjects", "test_subject",
    "n_train", "n_val", "n_test",
    "monitor", "best_score", "best_epoch", "epochs_run", "early_stopped",
    "n_params", "train_seconds",
    # metrics of the BEST checkpoint on the held-out subject
    "accuracy", "balanced_accuracy", "macro_f1", "auc", "cohen_kappa",
    "nll", "brier", "ece",
    # metrics of the LAST-epoch checkpoint on the same subject
    "last_accuracy", "last_balanced_accuracy", "last_macro_f1", "last_auc",
    "last_cohen_kappa",
    # metrics of the BEST checkpoint after test-time AdaBN (only when `eval.adabn: true`)
    "adabn_accuracy", "adabn_balanced_accuracy", "adabn_macro_f1", "adabn_auc",
    "adabn_cohen_kappa",
    "status", "error",
]


# --------------------------------------------------------------------------- #
# Normalization modes
# --------------------------------------------------------------------------- #
#: ``train.normalization`` grammar (2026-09-18, spiking/alignment round):
#:
#:   <trial_norm>                         e.g. "per_sample_zscore"   (unchanged behaviour)
#:   ea_session[+<trial_norm>]            Euclidean Alignment fitted on each SESSION's own
#:                                        trials (He & Wu, IEEE TBME 2020), then the optional
#:                                        per-trial normalization
#:   ea_subject[+<trial_norm>]            EA fitted on all pooled trials of a subject
#:   ea_session_mu / ea_subject_mu        same, but the reference covariance is computed on
#:                                        an 8-30 Hz copy of the trials (mu/beta band, where
#:                                        MI information lives) and applied to the broadband
#:                                        input; 0.5-40 Hz covariance is dominated by
#:                                        low-frequency power that EA would otherwise target
#:
#: After EA the aligned channels have unit *average* variance by construction, so a
#: channel-wise per-trial z-score would undo the whitening; use a per-trial **scalar** scale
#: (``per_sample_global``) or nothing after EA.
#:
#: EA is label-free and uses only the trials of the very recording it is applied to, so the
#: test subject is aligned with its **own unlabeled** data — the standard (transductive,
#: unsupervised) use of EA in cross-subject MI. No statistic crosses subjects or splits.
ALIGN_SCOPES = ("none", "session", "subject")
EA_REF_BAND_MU = (8.0, 30.0)


def split_normalization(mode: str) -> Tuple[str, str, Optional[Tuple[float, float]]]:
    """``"ea_session_mu+per_sample_global"`` -> ``("session", "per_sample_global", (8, 30))``.

    Plain modes map to ``("none", mode, None)``; a bare ``"ea_session"`` means EA only.
    """
    raw = (mode or "per_sample_zscore").strip().lower()
    head, _, tail = raw.partition("+")
    band = None
    if head.endswith("_mu"):
        head, band = head[:-3], EA_REF_BAND_MU
    if head in ("ea_session", "ea_subject"):
        return head.split("_", 1)[1], (tail or "none"), band
    if "+" in raw or band is not None:
        raise ValueError(f"unknown composite normalization '{mode}'")
    return "none", raw, None


#: Shrinkage of the reference covariance toward ``tr(R)/C * I`` before the inverse square root
#: (Ledoit-Wolf style). Plain EA (He & Wu 2020) uses 0; with 58 channels the smallest
#: eigen-directions of ``R`` are noise and un-shrunk EA amplifies them by orders of magnitude.
#: 0.1 is the value the Phase-2b alignment study used; it whitens every direction whose
#: variance is at least ~tr/C to within 10% and leaves genuinely tiny directions damped.
EA_SHRINKAGE = 0.1


def euclidean_align(X: np.ndarray, shrinkage: float = EA_SHRINKAGE,
                    ref_band: Optional[Tuple[float, float]] = None,
                    sfreq: float = 250.0) -> np.ndarray:
    """``R^{-1/2} X`` with ``R`` the mean trial covariance of ``X`` (``[n, C, T]`` -> same).

    With ``ref_band`` the reference covariance is estimated on a zero-phase FIR band-passed
    copy of the trials (the whitening matrix is then applied to the unfiltered input).
    """
    from ..methods.session_alignment import EuclideanAlignment

    if X.shape[0] < 2:
        raise ValueError("Euclidean Alignment needs at least 2 trials")
    ea = EuclideanAlignment(shrinkage=shrinkage)
    if ref_band is None:
        ea.fit(X)
    else:
        from scipy.signal import filtfilt, firwin

        lo, hi = float(ref_band[0]), float(ref_band[1])
        taps = firwin(101, [lo, hi], pass_zero=False, fs=float(sfreq))
        Xf = filtfilt(taps, [1.0], np.asarray(X, dtype=np.float64), axis=-1)
        ea.fit(Xf)
    return ea.transform(X).astype(np.float32)


@torch.no_grad()
def adapt_batchnorm_stats(model: torch.nn.Module, loader: DataLoader, *,
                          device: torch.device) -> int:
    """AdaBN (Li et al., Pattern Recognition 2018): replace every BatchNorm layer's running
    mean/var by the statistics of the data in ``loader`` (cumulative average over batches).

    Only BatchNorm modules are switched to train mode; dropout and everything else stay in
    eval mode, so the passes are deterministic and no gradient is taken. Labels are never
    read. Affine parameters are untouched. Returns the number of BatchNorm layers adapted
    (0 -> the model has none and nothing happened). The model is left in eval mode.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    momenta = [bn.momentum for bn in bns]
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative moving average == plain mean over all batches
        bn.train()
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        model(xb)
    for bn, mom in zip(bns, momenta):
        bn.momentum = mom
    model.eval()
    return len(bns)


# --------------------------------------------------------------------------- #
# Subject-level data
# --------------------------------------------------------------------------- #
@dataclass
class SubjectData:
    """All (pooled) trials of one subject, ready for training."""

    subject: str
    X: torch.Tensor                 # [n, C, T] float32, already per-trial normalized
    y: torch.Tensor                 # [n] int64 in {0,1}
    sessions: List[str]
    session_of_trial: List[str] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return int(self.y.numel())


def load_subject_data(
    records: Sequence[SessionRecord],
    *,
    n_channels: int,
    n_times: int,
    normalization: str = "per_sample_zscore",
    pool_sessions: bool = True,
    logger=None,
) -> "Dict[str, SubjectData]":
    """Load every ok session and pool it per subject (cross-subject needs subject blocks).

    Normalization is applied here because it is fit-free per trial: doing it once at load
    time costs nothing and guarantees train/val/test are treated identically.

    Raises if any session's shape disagrees with the configured ``n_channels`` /
    ``n_times`` — mixing datasets or channel counts must fail loudly, never silently.
    """
    if not pool_sessions:
        raise NotImplementedError(
            "pool_sessions=false (per-session cross-subject cells) is not implemented; "
            "the current protocol pools a subject's ok sessions."
        )
    align_scope, trial_norm, ref_band = split_normalization(normalization)
    by_subject = group_by_subject(records)
    out: "Dict[str, SubjectData]" = {}
    for subj, recs in by_subject.items():
        xs, ys, sess_of_trial = [], [], []
        for rec in recs:
            d = load_session_npz(rec.npz_path)
            X = np.ascontiguousarray(d["X"], dtype=np.float32)
            if X.ndim != 3 or X.shape[1] != n_channels or X.shape[2] != n_times:
                raise ValueError(
                    f"{rec.subject}/{rec.session}: expected [n,{n_channels},{n_times}], "
                    f"got {tuple(X.shape)} from {rec.npz_path}. Check data.n_channels / "
                    "data.n_times in the experiment config (datasets are never merged)."
                )
            if align_scope == "session":
                X = euclidean_align(X, ref_band=ref_band)
            xs.append(torch.from_numpy(X))
            ys.append(torch.from_numpy(normalize_labels(d["y"]).astype(np.int64)))
            sess_of_trial += [rec.session] * X.shape[0]
        X_cat = torch.cat(xs, dim=0)
        if align_scope == "subject":
            X_cat = torch.from_numpy(euclidean_align(X_cat.numpy(), ref_band=ref_band))
        X_all = normalize_trials(X_cat, trial_norm)
        y_all = torch.cat(ys, dim=0)
        out[subj] = SubjectData(subject=subj, X=X_all, y=y_all,
                                sessions=[r.session for r in recs],
                                session_of_trial=sess_of_trial)
        if logger is not None:
            logger.debug("loaded %s: %d trials over %d sessions", subj,
                         out[subj].n_trials, len(recs))
    if logger is not None:
        tot = sum(s.n_trials for s in out.values())
        logger.info("subject data ready: %d subjects / %d trials (normalization=%s)",
                    len(out), tot, normalization)
    return out


# --------------------------------------------------------------------------- #
# Subject-level folds
# --------------------------------------------------------------------------- #
@dataclass
class CrossSubjectFold:
    """One subject-level split.

    Under ``val_mode=subjects`` / ``trials``, train/val/test subject lists are pairwise
    disjoint. Under ``val_mode=sessions``, train and val share the same non-test subjects
    but restrict to disjoint session lists (``train_sessions`` / ``val_sessions``).
    """

    fold: int
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]
    train_sessions: Optional[List[str]] = None
    val_sessions: Optional[List[str]] = None

    def assert_disjoint(self) -> None:
        tr, va, te = set(self.train_subjects), set(self.val_subjects), set(self.test_subjects)
        if tr & te or va & te:
            raise AssertionError(
                f"fold {self.fold}: test subject leak "
                f"(train&test={sorted(tr & te)}, val&test={sorted(va & te)})"
            )
        session_val = self.train_sessions is not None or self.val_sessions is not None
        if session_val:
            tr_s = set(self.train_sessions or [])
            va_s = set(self.val_sessions or [])
            if not tr_s or not va_s:
                raise AssertionError(
                    f"fold {self.fold}: sessions val_mode requires non-empty "
                    "train_sessions and val_sessions"
                )
            if tr_s & va_s:
                raise AssertionError(
                    f"fold {self.fold}: train_sessions and val_sessions overlap "
                    f"({sorted(tr_s & va_s)})"
                )
            if tr != va:
                raise AssertionError(
                    f"fold {self.fold}: sessions val_mode expects identical train/val "
                    f"subject pools (train={sorted(tr)}, val={sorted(va)})"
                )
        elif tr & va:
            raise AssertionError(
                f"fold {self.fold}: subject sets overlap "
                f"(train&val={sorted(tr & va)})"
            )
        if not self.train_subjects or not self.test_subjects:
            raise AssertionError(f"fold {self.fold}: empty train or test subject list")


def _normalize_session_list(sessions: Optional[Sequence[str | int]]) -> Optional[List[str]]:
    if sessions is None:
        return None
    out = sorted({ses_id(s) for s in sessions})
    return out


def _pick_val_subjects(train_pool: List[str], *, n_val_subjects: Optional[int],
                       val_subject_fraction: float, rng: np.random.Generator) -> List[str]:
    if n_val_subjects is None:
        n_val = int(round(float(val_subject_fraction) * len(train_pool)))
    else:
        n_val = int(n_val_subjects)
    n_val = max(0, min(n_val, len(train_pool) - 1))
    if n_val == 0:
        return []
    chosen = rng.choice(np.asarray(train_pool), size=n_val, replace=False)
    return sorted(str(s) for s in chosen)


def make_subject_folds(
    subjects: Sequence[str],
    *,
    protocol: str = "kfold_subject",
    n_folds: int = 5,
    val_mode: str = "subjects",
    n_val_subjects: Optional[int] = None,
    val_subject_fraction: float = 0.15,
    split_seed: int = 0,
    test_subjects: Optional[Sequence[str]] = None,
    val_subjects: Optional[Sequence[str]] = None,
    test_fraction: float = 0.2,
    train_sessions: Optional[Sequence[str | int]] = None,
    val_sessions: Optional[Sequence[str | int]] = None,
) -> List[CrossSubjectFold]:
    """Build the subject-level folds for the chosen protocol (deterministic in ``split_seed``)."""
    protocol = (protocol or "kfold_subject").lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol '{protocol}' (use {list(PROTOCOLS)})")
    mode = (val_mode or "subjects").lower()
    if mode not in VAL_MODES:
        raise ValueError(f"unknown val_mode '{val_mode}' (use {list(VAL_MODES)})")
    subjects = sorted(set(str(s) for s in subjects))
    if len(subjects) < 3:
        raise ValueError(f"cross-subject needs >=3 subjects, got {len(subjects)}")
    rng = np.random.default_rng(int(split_seed))
    use_val_subjects = mode == "subjects"
    use_session_val = mode == "sessions"
    tr_sess = _normalize_session_list(train_sessions)
    va_sess = _normalize_session_list(val_sessions)
    if use_session_val:
        if tr_sess is None:
            tr_sess = list(DEFAULT_TRAIN_SESSIONS)
        if va_sess is None:
            va_sess = list(DEFAULT_VAL_SESSIONS)
        if set(tr_sess) & set(va_sess):
            raise ValueError(
                f"train_sessions and val_sessions must be disjoint "
                f"(got {tr_sess} vs {va_sess})"
            )

    def _fold(i: int, train: List[str], val: List[str], test: List[str]) -> CrossSubjectFold:
        return CrossSubjectFold(
            i, train, val, test,
            train_sessions=list(tr_sess) if use_session_val else None,
            val_sessions=list(va_sess) if use_session_val else None,
        )

    if protocol == "holdout":
        if test_subjects:
            test = sorted(str(s) for s in test_subjects)
            missing = sorted(set(test) - set(subjects))
            if missing:
                raise ValueError(f"holdout test_subjects not in the loaded set: {missing}")
        else:
            n_test = max(1, int(round(float(test_fraction) * len(subjects))))
            perm = rng.permutation(np.asarray(subjects))
            test = sorted(str(s) for s in perm[:n_test])
        pool = [s for s in subjects if s not in set(test)]
        if use_session_val:
            val = list(pool)
            train = list(pool)
        elif val_subjects:
            val = sorted(str(s) for s in val_subjects)
            bad = sorted(set(val) - set(pool))
            if bad:
                raise ValueError(f"holdout val_subjects must come from the train pool: {bad}")
            train = [s for s in pool if s not in set(val)]
        elif use_val_subjects:
            val = _pick_val_subjects(pool, n_val_subjects=n_val_subjects,
                                     val_subject_fraction=val_subject_fraction, rng=rng)
            train = [s for s in pool if s not in set(val)]
        else:
            val = []
            train = list(pool)
        folds = [_fold(0, train, val, test)]
    else:
        if protocol == "loso":
            groups = [[s] for s in subjects]
        else:
            k = int(n_folds)
            if k < 2 or k > len(subjects):
                raise ValueError(f"n_folds={k} invalid for {len(subjects)} subjects")
            perm = [str(s) for s in rng.permutation(np.asarray(subjects))]
            groups = [sorted(g.tolist()) for g in np.array_split(np.asarray(perm), k)]
        folds = []
        for i, test in enumerate(groups):
            pool = [s for s in subjects if s not in set(test)]
            if use_session_val:
                val = list(pool)
                train = list(pool)
            else:
                val = (_pick_val_subjects(pool, n_val_subjects=n_val_subjects,
                                          val_subject_fraction=val_subject_fraction, rng=rng)
                       if use_val_subjects else [])
                train = [s for s in pool if s not in set(val)]
            folds.append(_fold(i, train, val, sorted(test)))

    for f in folds:
        f.assert_disjoint()
    return folds


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _make_dataset(
    subject_data: "Dict[str, SubjectData]",
    subjects: Sequence[str],
    sessions: Optional[Sequence[str]] = None,
    adversarial: bool = False,
) -> Tuple[Optional[ConcatDataset], torch.Tensor]:
    """Concatenate subjects **without copying their trials**.

    ``torch.cat`` would duplicate the training split every cell — on WBCIC that is ~5.5 GB
    on top of the ~7 GB subject cache, per cell. ``ConcatDataset`` over per-subject
    ``TensorDataset``s gives the same shuffled stream with zero extra memory. Only the label
    vector is materialized (it is tiny, and stratification needs it).

    When ``sessions`` is set, only trials whose ``session_of_trial`` is in that set are kept
    (paper-aligned ``val_mode=sessions``).
    """
    if not subjects:
        return None, torch.empty(0, dtype=torch.long)
    sess_set = {ses_id(s) for s in sessions} if sessions is not None else None
    parts = []
    ys = []
    for s_idx, s in enumerate(subjects):
        sd = subject_data[s]
        if sess_set is None:
            if adversarial:
                sid = torch.full((sd.n_trials,), s_idx, dtype=torch.long)
                parts.append(TensorDataset(sd.X, sd.y, sid))
            else:
                parts.append(TensorDataset(sd.X, sd.y))
            ys.append(sd.y)
            continue
        idx = [i for i, sess in enumerate(sd.session_of_trial) if ses_id(sess) in sess_set]
        if not idx:
            continue
        if adversarial:
            sid = torch.full((len(idx),), s_idx, dtype=torch.long)
            parts.append(TensorDataset(sd.X[idx], sd.y[idx], sid))
        else:
            parts.append(TensorDataset(sd.X[idx], sd.y[idx]))
        ys.append(sd.y[idx])
    if not parts:
        return None, torch.empty(0, dtype=torch.long)
    y = torch.cat(ys, dim=0)
    return ConcatDataset(parts), y


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin: bool
            ) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin, drop_last=False)


def _trial_val_split(y: torch.Tensor, val_fraction: float, seed: int
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Stratified trial-level val slice (only used when ``val_mode='trials'``)."""
    idx = np.arange(int(y.numel()))
    if val_fraction <= 0:
        return idx, None
    from sklearn.model_selection import train_test_split
    labels = y.numpy()
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return idx, None
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=labels)
    return tr, va


def cell_signature(fold: CrossSubjectFold, *, dataset: str, model_name: str,
                   model_params: Dict, data_dims: Dict, spec: E2ESpec,
                   protocol: str, val_mode: str, seed: int) -> str:
    """Fingerprint of everything a resumed cell must not have changed.

    Guards the most dangerous silent failure of a resumable sweep: someone re-runs with a
    different ``--folds`` / ``--max-subjects`` / model params into the same output dir, and
    ``fold0`` now means a different set of subjects while a ``last.pt`` from the old split
    is still sitting there.

    The early-stopping settings are part of the fingerprint because they decide *which*
    epoch becomes ``best.pt``; see ``_stopping_identity`` for why ``max_epochs`` is included
    conditionally while ``early_stopping_patience`` always is.
    """
    payload = {
        "signature_version": SIGNATURE_VERSION,
        "dataset": dataset, "model": model_name, "protocol": protocol,
        "val_mode": val_mode, "seed": int(seed), "fold": fold.fold,
        "train_subjects": sorted(fold.train_subjects),
        "val_subjects": sorted(fold.val_subjects),
        "test_subjects": sorted(fold.test_subjects),
        "train_sessions": list(fold.train_sessions or []),
        "val_sessions": list(fold.val_sessions or []),
        "data_dims": {k: data_dims[k] for k in sorted(data_dims)},
        "model_params": {k: model_params[k] for k in sorted(model_params)},
        "normalization": spec.normalization,
        "monitor": spec.monitor, "monitor_mode": spec.monitor_mode,
        "optimizer": spec.optimizer, "lr": spec.lr, "weight_decay": spec.weight_decay,
        "batch_size": spec.batch_size, "scheduler": spec.scheduler,
        "grad_clip_norm": spec.grad_clip_norm, "val_fraction": spec.val_fraction,
        **_stopping_identity(spec),
        # Accumulation changes BatchNorm statistics, so it is part of the cell's identity.
        "micro_batch_size": spec.micro_batch_size,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _val_evaluator():
    """Validation metrics for model selection: full metric set + val loss."""

    def _fn(model, loader, device) -> Dict[str, float]:
        y_true, y_pred, probs, _ = predict(model, loader, device=device)
        metrics = evaluate_predictions(y_true, y_pred, probs)
        eps = 1e-12
        p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
        metrics["val_loss"] = float(-np.log(p).mean())
        return {k: float(v) for k, v in metrics.items()}

    return _fn


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
def run_cell(
    subject_data: "Dict[str, SubjectData]",
    fold: CrossSubjectFold,
    *,
    dataset: str,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: E2ESpec,
    seed: int,
    device: torch.device,
    ckpt_dir: Path,
    result_json: Path,
    protocol: str = "kfold_subject",
    val_mode: str = "subjects",
    split_seed: int = 0,
    resume: bool = True,
    logger=None,
) -> List[Dict[str, object]]:
    """Train + evaluate one (model, fold, seed) cell; return one row per test subject.

    ``split_seed`` is recorded in every row and in ``result.json``. Without it the per-cell
    record cannot tell which subject partition produced it, and a summarizer reading
    ``result.json`` (the authoritative record) has no way to compute the spread across
    partitions — which is the whole point of sweeping ``split_seeds``.
    """
    fold.assert_disjoint()
    paths = CellPaths(ckpt_dir=Path(ckpt_dir), result_json=Path(result_json))
    # Checkpoints live under ckpt_dir, often a different tree. np.savez_compressed
    # does not mkdir, so the cell dir (result.json / predictions.npz) must exist first.
    paths.result_json.parent.mkdir(parents=True, exist_ok=True)
    signature = cell_signature(fold, dataset=dataset, model_name=model_name,
                               model_params=model_params, data_dims=data_dims, spec=spec,
                               protocol=protocol, val_mode=val_mode, seed=seed)
    if resume and paths.is_complete():
        with open(paths.result_json, "r", encoding="utf-8") as f:
            done = json.load(f)
        stored_version = int(done.get("signature_version", 1))
        if stored_version != SIGNATURE_VERSION:
            raise RuntimeError(
                f"{paths.result_json} was written with cell-signature v{stored_version}, "
                f"but this code computes v{SIGNATURE_VERSION}. The payload changed, so the "
                "two are not comparable — v2 added max_epochs and early_stopping_patience, "
                "which v1 let you change without invalidating a cell. Historical runs are "
                "read-only (AGENTS.md 'do not overwrite completed *_v1'); use a fresh --out."
            )
        stored = done.get("cell_signature")
        if stored is not None and stored != signature:
            raise RuntimeError(
                f"{paths.result_json} was produced under a different configuration "
                f"(stored signature {stored}, current {signature}). Use a fresh --out for the "
                "new setting, or --no-resume to overwrite."
            )
        if logger is not None:
            logger.info("skip completed cell: %s", paths.result_json)
        return list(done.get("rows") or [])

    set_seed(int(seed))
    pin = device.type == "cuda"
    mode = (val_mode or "subjects").lower()
    is_adversarial = "adv" in (model_name or "").lower()
    train_ds, ytr = _make_dataset(
        subject_data, fold.train_subjects, sessions=fold.train_sessions,
        adversarial=is_adversarial,
    )

    # Under gradient accumulation the train loader yields micro-batches; the trainer groups
    # them so the optimizer still sees `spec.batch_size`. Eval loaders keep the full batch.
    micro_bs = int(spec.micro_batch_size or spec.batch_size)

    if mode in ("subjects", "sessions"):
        val_ds, yva = _make_dataset(
            subject_data, fold.val_subjects, sessions=fold.val_sessions,
        )
        train_loader = _loader(train_ds, batch_size=micro_bs, shuffle=True,
                               num_workers=spec.num_workers, pin=pin)
        val_loader = (None if val_ds is None else
                      _loader(val_ds, batch_size=spec.batch_size, shuffle=False,
                              num_workers=spec.num_workers, pin=pin))
        n_train, n_val = int(ytr.numel()), int(yva.numel())
    else:
        tr_idx, va_idx = _trial_val_split(ytr, spec.val_fraction, int(seed))
        train_loader = _loader(Subset(train_ds, tr_idx.tolist()), batch_size=micro_bs,
                               shuffle=True, num_workers=spec.num_workers, pin=pin)
        val_loader = (None if va_idx is None else
                      _loader(Subset(train_ds, va_idx.tolist()), batch_size=spec.batch_size,
                              shuffle=False, num_workers=spec.num_workers, pin=pin))
        n_train = int(len(tr_idx))
        n_val = 0 if va_idx is None else int(len(va_idx))

    # Monitoring-only loaders for the per-epoch train/val/test curves. The test loader is
    # handed to the trainer for *plotting only*; `code/training/e2e_trainer.py` computes
    # best.pt / early stopping from the val loader alone (leakage guard lives there).
    train_eval_loader = test_curve_loader = None
    if getattr(spec, "curves", False):
        cap = int(getattr(spec, "train_eval_max_trials", 2000) or 0)
        n_tr = int(ytr.numel())
        if cap > 0 and n_tr > cap:
            # Deterministic subset so the curve is comparable across epochs and resumes.
            sub = np.linspace(0, n_tr - 1, num=cap, dtype=int).tolist()
            train_eval_ds = Subset(train_ds, sub)
        else:
            train_eval_ds = train_ds
        train_eval_loader = _loader(train_eval_ds, batch_size=spec.batch_size, shuffle=False,
                                    num_workers=spec.num_workers, pin=pin)
        test_ds_all, _ = _make_dataset(subject_data, fold.test_subjects)
        if test_ds_all is not None:
            test_curve_loader = _loader(test_ds_all, batch_size=spec.batch_size, shuffle=False,
                                        num_workers=spec.num_workers, pin=pin)

    model = build_model(model_name, params=dict(model_params), **data_dims)
    desc = model.describe() if hasattr(model, "describe") else {}
    if logger is not None:
        logger.info("[cell] %s fold=%d seed=%d | train=%d subj/%d trials val=%d subj/%d trials "
                    "test=%s | train_sess=%s val_sess=%s | params=%s",
                    model_name, fold.fold, seed, len(fold.train_subjects), n_train,
                    len(fold.val_subjects), n_val, ",".join(fold.test_subjects),
                    fold.train_sessions, fold.val_sessions, desc.get("n_params"))

    info = train_end_to_end(
        model, train_loader, val_loader, spec=spec, device=device,
        ckpt_dir=paths.ckpt_dir, resume=resume, evaluate_fn=_val_evaluator(),
        epoch_seed_base=int(seed) * 100_000 + int(fold.fold) * 1_000,
        run_signature=signature,
        train_eval_loader=train_eval_loader, test_loader=test_curve_loader,
        logger=logger,
    )

    # Score BOTH checkpoints on each held-out subject separately (mean +/- std over
    # subjects is the headline number; pooled metrics go into result.json).
    rows: List[Dict[str, object]] = []
    per_subject_detail: List[Dict[str, object]] = []
    last_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_ck = paths.best
    best_state = (torch.load(best_ck, map_location=device, weights_only=False)["model_state"]
                  if best_ck.exists() else None)

    pred_dump: Dict[str, List] = {"subject": [], "session": [], "y_true": [],
                                  "y_pred_best": [], "y_pred_last": [],
                                  "probs_best": [], "probs_last": []}

    # Spiking models expose eval-time firing statistics (per-layer rates -> energy). They
    # accumulate over every eval forward, so drain them right before the held-out subject's
    # prediction pass and read them right after; the result is per-subject, best checkpoint.
    spike_stats_fn = getattr(model, "pop_spike_stats", None)
    if not callable(spike_stats_fn):
        spike_stats_fn = None

    for subj in fold.test_subjects:
        sd = subject_data[subj]
        test_ds, _ = _make_dataset(subject_data, [subj])
        test_loader = _loader(test_ds, batch_size=spec.batch_size, shuffle=False,
                              num_workers=spec.num_workers, pin=pin)
        if best_state is not None:
            model.load_state_dict(best_state)
        if spike_stats_fn is not None:
            spike_stats_fn()                                # drain val-time counters
        y_true, y_pred, probs, _ = predict(model, test_loader, device=device)
        best_metrics = evaluate_predictions(y_true, y_pred, probs)
        spike_stats = spike_stats_fn() if spike_stats_fn is not None else None

        model.load_state_dict(last_state)
        y_true_l, y_pred_l, probs_l, _ = predict(model, test_loader, device=device)
        last_metrics = evaluate_predictions(y_true_l, y_pred_l, probs_l)

        # Optional AdaBN scoring of the BEST checkpoint: BatchNorm running statistics are
        # re-estimated on this subject's unlabeled trials (labels never read), then scored.
        adabn_metrics = None
        adabn_spike_stats = None
        if getattr(spec, "adabn", False) and best_state is not None:
            model.load_state_dict(best_state)
            n_bn = adapt_batchnorm_stats(model, test_loader, device=device)
            if n_bn > 0:
                if spike_stats_fn is not None:
                    spike_stats_fn()
                y_true_a, y_pred_a, probs_a, _ = predict(model, test_loader, device=device)
                adabn_metrics = evaluate_predictions(y_true_a, y_pred_a, probs_a)
                adabn_spike_stats = spike_stats_fn() if spike_stats_fn is not None else None
                pred_dump.setdefault("y_pred_adabn", []).append(np.asarray(y_pred_a, dtype=np.int16))
                pred_dump.setdefault("probs_adabn", []).append(np.asarray(probs_a, dtype=np.float32))

        # `_make_dataset` preserves subject-trial order and the loader is unshuffled, so
        # these arrays align element-wise with `sd.session_of_trial`.
        n = int(np.asarray(y_true).size)
        pred_dump["subject"].append(np.full(n, subj, dtype="<U32"))
        pred_dump["session"].append(np.asarray(sd.session_of_trial[:n], dtype="<U32")
                                    if sd.session_of_trial else np.full(n, "", dtype="<U32"))
        pred_dump["y_true"].append(np.asarray(y_true, dtype=np.int16))
        pred_dump["y_pred_best"].append(np.asarray(y_pred, dtype=np.int16))
        pred_dump["y_pred_last"].append(np.asarray(y_pred_l, dtype=np.int16))
        pred_dump["probs_best"].append(np.asarray(probs, dtype=np.float32))
        pred_dump["probs_last"].append(np.asarray(probs_l, dtype=np.float32))

        rows.append({
            "dataset": dataset, "model": model_name, "protocol": protocol,
            "fold": fold.fold, "seed": int(seed), "split_seed": int(split_seed),
            "n_train_subjects": len(fold.train_subjects),
            "n_val_subjects": len(fold.val_subjects), "test_subject": subj,
            "n_train": n_train, "n_val": n_val, "n_test": sd.n_trials,
            "monitor": info["monitor"], "best_score": float(info["best_score"]),
            "best_epoch": int(info["best_epoch"]), "epochs_run": int(info["epochs_run"]),
            "early_stopped": bool(info["early_stopped"]),
            "n_params": int(desc.get("n_params", 0)),
            "train_seconds": float(info["train_seconds"]),
            **{k: float(v) for k, v in best_metrics.items()},
            "last_accuracy": float(last_metrics["accuracy"]),
            "last_balanced_accuracy": float(last_metrics["balanced_accuracy"]),
            "last_macro_f1": float(last_metrics["macro_f1"]),
            "last_auc": float(last_metrics["auc"]),
            "last_cohen_kappa": float(last_metrics["cohen_kappa"]),
            **({f"adabn_{k}": float(adabn_metrics[k]) for k in
                ("accuracy", "balanced_accuracy", "macro_f1", "auc", "cohen_kappa")}
               if adabn_metrics is not None else {}),
            "status": "ok", "error": "",
        })
        detail = {"test_subject": subj, "n_trials": sd.n_trials,
                  "sessions": sd.sessions,
                  "best": {k: float(v) for k, v in best_metrics.items()},
                  "last": {k: float(v) for k, v in last_metrics.items()}}
        if adabn_metrics is not None:
            detail["adabn"] = {k: float(v) for k, v in adabn_metrics.items()}
        if spike_stats is not None:
            detail["spike_stats"] = spike_stats
        if adabn_spike_stats is not None:
            detail["spike_stats_adabn"] = adabn_spike_stats
        per_subject_detail.append(detail)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Per-trial predictions, as a compressed npz beside result.json (~100 kB/cell).
    #
    # These CANNOT be recovered later without retraining, and they are what every
    # trial-level analysis needs: McNemar tests between models, confusion matrices,
    # post-hoc ensembling, and per-session breakdowns of a test subject (WBCIC-SHU's
    # within-session accuracy rises across sessions because subjects *learn* MI — see
    # 4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md §2.4 — so session labels matter).
    #
    # Note the *subject-level* paired tests that decide "model A beats model B" do not need
    # this file: those run on the per-subject rows above. This is for the trial-level layer.
    pred_path = Path(paths.result_json).with_name("predictions.npz")
    np.savez_compressed(
        pred_path,
        **{k: np.concatenate(v, axis=0) for k, v in pred_dump.items() if v},
        classes=np.arange(int(data_dims["n_classes"]), dtype=np.int16),
    )

    save_json({
        "dataset": dataset, "model": model_name, "fold": asdict(fold), "seed": int(seed),
        "split_seed": int(split_seed),
        "protocol": protocol, "cell_signature": signature,
        "signature_version": SIGNATURE_VERSION, "data_dims": data_dims,
        "model_params": model_params, "model_desc": desc, "spec": asdict(spec),
        "val_mode": val_mode,
        "train_info": {k: v for k, v in info.items() if k != "history"},
        "history": info["history"], "per_subject": per_subject_detail, "rows": rows,
        "checkpoints": {"best": str(paths.best), "last": str(paths.last)},
        "predictions": str(pred_path),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, paths.result_json)
    return rows


# --------------------------------------------------------------------------- #
# Sweep over models x folds x seeds
# --------------------------------------------------------------------------- #
def run_cross_subject(
    records: Sequence[SessionRecord],
    *,
    dataset: str,
    models: Sequence[str],
    model_params_all: Dict[str, Dict],
    data_dims: Dict,
    spec: E2ESpec,
    seeds: Sequence[int],
    device: torch.device,
    protocol: str = "kfold_subject",
    n_folds: int = 5,
    folds_subset: Optional[Sequence[int]] = None,
    val_mode: str = "subjects",
    n_val_subjects: Optional[int] = None,
    val_subject_fraction: float = 0.15,
    split_seed: int = 0,
    split_seeds: Optional[Sequence[int]] = None,
    test_subjects: Optional[Sequence[str]] = None,
    val_subjects: Optional[Sequence[str]] = None,
    test_fraction: float = 0.2,
    train_sessions: Optional[Sequence[str | int]] = None,
    val_sessions: Optional[Sequence[str | int]] = None,
    normalization: str = "per_sample_zscore",
    micro_batch_per_model: Optional[Dict[str, int]] = None,
    out_dir: Path,
    ckpt_root: Optional[Path] = None,
    resume: bool = True,
    logger=None,
) -> Dict[str, object]:
    """Run the whole cross-subject sweep, resuming whatever is already on disk.

    ``split_seeds`` re-draws the subject partition: each value produces its own set of
    folds, so ``split_seeds=[41..45]`` with ``seeds=[0]`` means "5 different 5-fold
    partitions, identical training seed" — the variance reported is then the variance of
    the split, not of the initialization. Cell tags carry the split seed, so the
    partitions never collide in one output directory.
    """
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"
    cells_dir = out_dir / "cells"
    splits_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    subject_data = load_subject_data(
        records, n_channels=data_dims["n_channels"], n_times=data_dims["n_times"],
        normalization=normalization, logger=logger,
    )

    split_seed_list = [int(s) for s in (split_seeds if split_seeds else [split_seed])]
    multi_split = len(split_seed_list) > 1
    folds_by_split: "Dict[int, List[CrossSubjectFold]]" = {}
    for ss in split_seed_list:
        folds = make_subject_folds(
            list(subject_data), protocol=protocol, n_folds=n_folds, val_mode=val_mode,
            n_val_subjects=n_val_subjects, val_subject_fraction=val_subject_fraction,
            split_seed=ss, test_subjects=test_subjects, val_subjects=val_subjects,
            test_fraction=test_fraction, train_sessions=train_sessions,
            val_sessions=val_sessions,
        )
        folds_by_split[ss] = folds
        save_json({
            "dataset": dataset, "protocol": protocol, "n_folds": len(folds),
            "val_mode": val_mode, "split_seed": ss,
            "train_sessions": list(folds[0].train_sessions or []) if folds else [],
            "val_sessions": list(folds[0].val_sessions or []) if folds else [],
            "n_subjects": len(subject_data), "normalization": normalization,
            "subjects": sorted(subject_data),
            "trials_per_subject": {s: subject_data[s].n_trials for s in sorted(subject_data)},
            "folds": [asdict(f) for f in folds],
        }, splits_dir / f"cross_subject_folds__{protocol}__seed{ss}.json")

    n_total_folds = sum(len(f) for f in folds_by_split.values())
    if logger is not None:
        logger.info("cross-subject plan: dataset=%s protocol=%s val_mode=%s "
                    "split_seeds=%s folds/split=%d models=%s train_seeds=%s",
                    dataset, protocol, val_mode, split_seed_list,
                    len(folds_by_split[split_seed_list[0]]), list(models), list(seeds))

    all_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for model_name in models:
        mp = dict((model_params_all or {}).get(model_name, {}))
        # Models whose activations do not fit one full batch train with gradient
        # accumulation, so the optimizer still sees the recipe's `batch_size`.
        micro_bs = (micro_batch_per_model or {}).get(model_name)
        spec_m = replace(spec, micro_batch_size=int(micro_bs)) if micro_bs else spec
        if micro_bs and logger is not None:
            logger.info("%s: micro_batch=%d x accum=%d -> effective batch %d",
                        model_name, int(micro_bs), spec_m.accum_steps, spec_m.batch_size)
        for ss in split_seed_list:
            selected = [f for f in folds_by_split[ss]
                        if folds_subset is None or f.fold in set(int(i) for i in folds_subset)]
            for fold in selected:
                for seed in seeds:
                    # Include the split seed in the tag whenever more than one split is
                    # swept, or the split is not the legacy default 0. 3C LOSO at seed 0
                    # (running jobs; paper_baseline_3c_821_v1) keeps `model__foldK__seedS`.
                    # A lone non-zero seed (2C/SHU yaml `split_seeds: [43]`) must keep
                    # `split43` in the tag (D-030 / D-036).
                    # `safe_tag` turns an alias `atcnet@tcn8` into `atcnet-tcn8` for paths;
                    # the logical name (with `@`) stays in rows and the cell signature.
                    split_part = f"split{ss}__" if multi_split or int(ss) != 0 else ""
                    tag = f"{safe_tag(model_name)}__{split_part}fold{fold.fold}__seed{int(seed)}"
                    ck = ((Path(ckpt_root) / tag) if ckpt_root is not None
                          else (cells_dir / tag / "ckpt"))
                    try:
                        rows = run_cell(
                            subject_data, fold, dataset=dataset, model_name=model_name,
                            model_params=mp, data_dims=data_dims, spec=spec_m,
                            seed=int(seed), device=device, ckpt_dir=ck,
                            result_json=cells_dir / tag / "result.json",
                            protocol=protocol, val_mode=val_mode, split_seed=ss,
                            resume=resume, logger=logger,
                        )
                        for r in rows:
                            r.setdefault("split_seed", ss)   # legacy cells lack the key
                        all_rows += rows
                    except Exception as exc:  # keep the sweep alive; record and continue
                        msg = f"{type(exc).__name__}: {exc}"
                        failures.append({"cell": tag, "error": msg})
                        if logger is not None:
                            logger.exception("cell FAILED %s: %s", tag, msg)
                        all_rows.append({
                            "dataset": dataset, "model": model_name, "protocol": protocol,
                            "fold": fold.fold, "seed": int(seed), "split_seed": ss,
                            "n_train_subjects": len(fold.train_subjects),
                            "n_val_subjects": len(fold.val_subjects),
                            "test_subject": ",".join(fold.test_subjects),
                            "status": "failed", "error": msg,
                        })
    return {
        "rows": all_rows,
        "folds": [asdict(f) for ss in split_seed_list for f in folds_by_split[ss]],
        "failures": failures,
        "split_seeds": split_seed_list, "n_folds_total": n_total_folds,
        "n_subjects": len(subject_data), "n_cells_ok": sum(1 for r in all_rows
                                                           if r.get("status") == "ok"),
        "n_cells_failed": len(failures),
        "checkpoint_names": {"best": BEST_CKPT, "last": LAST_CKPT},
    }
