"""Bangun fitur dari rekaman CHB-MIT, lalu latih dan evaluasi Random Forest
dengan leave-one-recording-out (hanya rekaman yang berisi kejang yang jadi data uji).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from download_subset import parse_summary
from features import extract, feature_names, label_windows

ROOT = Path(__file__).resolve().parents[1]
WIN, STEP = 4.0, 2.0


def load_recording(path: Path):
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick([c for c in raw.ch_names if c not in ("-", "--0", "--1", "--2", "--3", "--4")])
    raw.filter(0.5, 60.0, verbose="ERROR")
    data = raw.get_data().astype(np.float32) * 1e6  # volt -> mikrovolt
    return data, raw.info["sfreq"], raw.ch_names


def build_features(patient: str) -> pd.DataFrame:
    cache = ROOT / "outputs" / f"{patient}_features.pkl"
    if cache.exists():
        print(f"pakai cache fitur: {cache.name}")
        return pd.read_pickle(cache)

    data_dir = ROOT / "data" / patient
    seizures = parse_summary((data_dir / f"{patient}-summary.txt").read_text())
    frames, ref_channels = [], None
    for edf in sorted(data_dir.glob("*.edf")):
        print(f"memproses {edf.name} ...")
        sig, fs, chs = load_recording(edf)
        if ref_channels is None:
            ref_channels = chs
        if chs != ref_channels:  # beberapa rekaman CHB-MIT punya montase berbeda
            print(f"  lewati: susunan kanal berbeda")
            continue
        X, t0 = extract(sig, fs, WIN, STEP)
        y = label_windows(t0, WIN, seizures.get(edf.name, []))
        df = pd.DataFrame(X, columns=feature_names(chs))
        df["recording"], df["t_start"], df["label"] = edf.name, t0, y
        frames.append(df)
        del sig  # bebaskan RAM sebelum file berikutnya
    out = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(exist_ok=True)
    out.to_pickle(cache)
    return out


def alarm_events(pred: np.ndarray, label: np.ndarray) -> dict:
    """Gabungkan jendela positif berurutan menjadi satu alarm.
    Alarm yang tidak menyentuh jendela kejang dihitung sebagai alarm palsu."""
    hit, false, i = False, 0, 0
    while i < len(pred):
        if pred[i]:
            j = i
            while j < len(pred) and pred[j]:
                j += 1
            if label[i:j].any():
                hit = True
            else:
                false += 1
            i = j
        else:
            i += 1
    return {"hit": hit, "false": false}


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if "|" in c]
    rows, importances = [], []
    test_recs = df["recording"].unique()  # semua rekaman, termasuk yang tanpa kejang
    for rec in test_recs:
        train, test = df[df.recording != rec], df[df.recording == rec]
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     min_samples_leaf=2, n_jobs=-1, random_state=0)
        clf.fit(train[feat_cols], train.label)
        prob = clf.predict_proba(test[feat_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test.label, pred, labels=[0, 1]).ravel()
        hours = len(test) * STEP / 3600
        has_sz = test.label.sum() > 0
        events = alarm_events(pred, test.label.to_numpy())
        rows.append({
            "recording": rec,
            "has_seizure": bool(has_sz),
            "seizure_detected": bool(events["hit"]) if has_sz else np.nan,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": f1_score(test.label, pred, zero_division=0) if has_sz else np.nan,
            "pr_auc": average_precision_score(test.label, prob) if has_sz else np.nan,
            "false_alarm_windows_per_hour": fp / hours,
            "false_alarm_events_per_hour": events["false"] / hours,
        })
        importances.append(clf.feature_importances_)
        plot_timeline(test, prob, rec)

    imp = pd.Series(np.mean(importances, axis=0), index=feat_cols).sort_values(ascending=False)
    imp.head(20).to_csv(ROOT / "outputs" / "top_features.csv")
    ax = imp.head(15)[::-1].plot.barh(figsize=(7, 5), title="Feature importance (rata-rata fold)")
    ax.figure.tight_layout()
    ax.figure.savefig(ROOT / "outputs" / "feature_importance.png", dpi=120)
    plt.close(ax.figure)
    return pd.DataFrame(rows)


def plot_timeline(test: pd.DataFrame, prob: np.ndarray, rec: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(test.t_start / 60, prob, lw=0.8, label="probabilitas kejang")
    ax.fill_between(test.t_start / 60, 0, test.label, color="red", alpha=0.25, label="kejang (anotasi)")
    ax.set(xlabel="menit", ylim=(0, 1.05), title=rec)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"timeline_{rec.replace('.edf', '')}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="chb01")
    args = p.parse_args()

    df = build_features(args.patient)
    print(f"{len(df)} jendela, {int(df.label.sum())} berlabel kejang "
          f"({df.label.mean():.2%}), {df.recording.nunique()} rekaman")
    res = evaluate(df)
    res.to_csv(ROOT / "outputs" / "metrics_per_recording.csv", index=False)
    print(res.round(3).to_string(index=False))
    for name, grp in res.groupby("has_seizure"):
        print(f"\nrata-rata, rekaman {'dengan' if name else 'tanpa'} kejang:")
        print(grp.drop(columns=["recording", "has_seizure"]).astype(float).mean().round(3).to_string())


if __name__ == "__main__":
    main()
