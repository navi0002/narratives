import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import signal
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def find_bold_file(bids_root: str, subject: str, task: str, run: Optional[str] = None) -> str:
    """
    Locate a BIDS fMRI file for a given subject/task/(optional) run.

    Parameters
    ----------
    bids_root : str
        Path to BIDS root directory.
    subject : str
        Subject ID in BIDS form (e.g., "sub-001").
    task : str
        Task label (e.g., "pieman").
    run : Optional[str]
        Optional run label without the "run-" prefix (e.g., "1").

    Returns
    -------
    str
        Absolute path to the first matching BOLD NIfTI file.
    """
    func_dir = os.path.join(bids_root, subject, "func")
    if run is None:
        pattern = f"{subject}_task-{task}_*bold.nii.gz"
    else:
        pattern = f"{subject}_task-{task}_run-{run}_*bold.nii.gz"
    matches = sorted(glob(os.path.join(func_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No BOLD file found for {subject} task {task} run {run}")
    return matches[0]


def load_events(bids_root: str, subject: str, task: str, run: Optional[str] = None) -> pd.DataFrame:
    """
    Load events.tsv for the scan; required for onset/duration and optional story timing.
    """
    func_dir = os.path.join(bids_root, subject, "func")
    if run is None:
        pattern = f"{subject}_task-{task}_*events.tsv"
    else:
        pattern = f"{subject}_task-{task}_run-{run}_*events.tsv"
    matches = sorted(glob(os.path.join(func_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No events.tsv found for {subject} task {task} run {run}")
    events = pd.read_csv(matches[0], sep="\t")
    return events


def load_alignment_gentle(align_json_path: str) -> pd.DataFrame:
    """
    Load Gentle alignment JSON with per-word timing.

    Returns a DataFrame with columns: [word, start, end].
    """
    with open(align_json_path) as f:
        data = json.load(f)
    words = []
    for w in data.get("words", []):
        if w.get("case") == "success" and "start" in w and "end" in w:
            words.append({"word": w.get("alignedWord", w.get("word", "")),
                          "start": float(w["start"]),
                          "end": float(w["end"])})
    return pd.DataFrame(words)


def simple_uniform_alignment(transcript: str, duration_s: float) -> pd.DataFrame:
    """
    Fallback alignment: distribute tokens uniformly across the given duration.
    """
    raw_tokens = [t for t in transcript.strip().split() if t]
    if len(raw_tokens) == 0:
        return pd.DataFrame(columns=["word", "start", "end"])  
    token_dur = duration_s / len(raw_tokens)
    rows = []
    for i, tok in enumerate(raw_tokens):
        start = i * token_dur
        end = start + token_dur
        rows.append({"word": tok, "start": start, "end": end})
    return pd.DataFrame(rows)


def build_text_chunks(words_df: pd.DataFrame, chunk_s: float) -> List[Tuple[float, float, str]]:
    """
    Aggregate words into fixed-width time chunks.
    Returns list of (chunk_start, chunk_end, text).
    """
    if words_df.empty:
        return []
    t0 = float(words_df["start"].min())
    t1 = float(words_df["end"].max())
    chunks = []
    cur = t0
    while cur < t1:
        nxt = min(cur + chunk_s, t1)
        in_chunk = words_df[(words_df["start"] < nxt) & (words_df["end"] > cur)]
        text = " ".join(in_chunk["word"].tolist())
        chunks.append((cur, nxt, text))
        cur = nxt
    return chunks


def load_emotion_pipeline(model_name: str = "cardiffnlp/twitter-roberta-base-emotion"):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, return_all_scores=True, truncation=True)


def compute_emotion_features(chunks: List[Tuple[float, float, str]], emo_pipe) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute emotion probabilities for each chunk.

    Returns
    -------
    features : ndarray [num_chunks, num_emotions]
    labels : list of emotion labels
    centers_s : ndarray [num_chunks] center time in seconds for each chunk
    """
    if len(chunks) == 0:
        return np.zeros((0, 0)), [], np.zeros((0,))
    texts = [t[2] if t[2] else "" for t in chunks]
    outputs = emo_pipe(texts)
    # Determine label order from first output
    labels = [d["label"] for d in outputs[0]] if outputs and outputs[0] else []
    feats = np.zeros((len(chunks), len(labels)), dtype=np.float32)
    for i, out in enumerate(outputs):
        score_by_label = {d["label"]: float(d["score"]) for d in out}
        feats[i] = [score_by_label.get(lbl, 0.0) for lbl in labels]
    centers = np.array([(s + e) / 2.0 for (s, e, _) in chunks], dtype=np.float32)
    return feats, labels, centers


def compute_emotion_features_mock(chunks: List[Tuple[float, float, str]]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Lightweight heuristic emotion features without transformers/torch.
    Uses simple lexical cues to produce 6 emotions and normalizes to probabilities.
    """
    labels = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
    if len(chunks) == 0:
        return np.zeros((0, len(labels))), labels, np.zeros((0,))

    positive_words = {"love", "happy", "great", "good", "joy", "like", "win"}
    negative_words = {"bad", "sad", "angry", "hate", "fear", "worry", "lose"}
    surprise_words = {"wow", "suddenly", "unexpected", "surprise"}
    disgust_words = {"gross", "disgust", "yuck"}

    feats = np.zeros((len(chunks), len(labels)), dtype=np.float32)
    centers = np.array([(s + e) / 2.0 for (s, e, _) in chunks], dtype=np.float32)
    for i, (_, _, text) in enumerate(chunks):
        tokens = set((text or "").lower().split())
        joy = len(tokens & positive_words)
        sadness = 1 if ("sad" in tokens or "cry" in tokens) else 0
        anger = 1 if ("angry" in tokens or "anger" in tokens or "mad" in tokens or "hate" in tokens) else 0
        fear = 1 if ("fear" in tokens or "scared" in tokens or "worry" in tokens) else 0
        surprise = len(tokens & surprise_words)
        disgust = len(tokens & disgust_words)
        vec = np.array([joy, sadness, anger, fear, surprise, disgust], dtype=np.float32)
        if vec.sum() == 0:
            vec = np.ones_like(vec) / len(vec)
        else:
            vec = vec / vec.sum()
        feats[i] = vec
    return feats, labels, centers


def hrf_spm(tr: float, oversampling: int = 16, time_length: float = 32.0, onset: float = 0.0) -> np.ndarray:
    """
    Approximate SPM HRF sampled at TR. Based on two gamma functions.
    """
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)
    # Parameters (SPM canonical):
    peak1 = signal.gamma.pdf(time_stamps, 6)
    peak2 = signal.gamma.pdf(time_stamps, 12)
    hrf = peak1 - 0.35 * peak2
    hrf /= np.sum(hrf)
    # Downsample to TR grid
    down_idx = np.arange(int(onset / dt), len(hrf), oversampling)
    return hrf[down_idx[:int(time_length / tr)]]


def resample_features_to_tr(feature_times_s: np.ndarray, features: np.ndarray, tr: float, n_trs: int, start_offset_s: float = 0.0) -> np.ndarray:
    """
    Resample sparse feature timeseries to TR grid via nearest assignment and optional HRF convolution.
    """
    if features.size == 0 or n_trs <= 0:
        return np.zeros((n_trs, 0), dtype=np.float32)
    t_grid = start_offset_s + np.arange(n_trs) * tr
    # Nearest-neighbor assignment
    idx = np.searchsorted(feature_times_s, t_grid, side="left")
    idx = np.clip(idx, 0, len(feature_times_s) - 1)
    X = features[idx]
    return X


def build_fir_design(X: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Create a simple FIR-lagged design matrix with lags [0..n_lags-1] TRs.
    """
    if X.size == 0:
        return X
    T, F = X.shape
    cols = []
    for lag in range(n_lags):
        pad = np.zeros((lag, F), dtype=X.dtype)
        cols.append(np.vstack([pad, X[:T - lag]]))
    return np.hstack(cols)


def make_brain_mask(nimg: nib.Nifti1Image, min_tsnr: float = 0.0) -> np.ndarray:
    data = nimg.get_fdata()
    if data.ndim != 4:
        raise ValueError("Expected 4D fMRI data")
    mean_img = np.nanmean(data, axis=-1)
    mask = np.isfinite(mean_img) & (mean_img > 0)
    return mask


def fit_ridge_encoding(X: np.ndarray, Y: np.ndarray, alphas: List[float], n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit RidgeCV voxel-wise; return predictions and coefficients.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    # KFold for predict-eval split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    Y_pred = np.zeros_like(Y)

    for train_idx, test_idx in kf.split(Xz):
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(Xz[train_idx], Y[train_idx])
        Y_pred[test_idx] = model.predict(Xz[test_idx])

    # Fit on full data for coefficients (optional)
    model_final = RidgeCV(alphas=alphas, cv=5)
    model_final.fit(Xz, Y)
    coefs = model_final.coef_
    return Y_pred, coefs


def corrcoef_timewise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Pearson r between columns of A and B across time.
    Returns 1D array of length n_cols.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A -= A.mean(axis=0, keepdims=True)
    B -= B.mean(axis=0, keepdims=True)
    num = np.sum(A * B, axis=0)
    den = np.sqrt(np.sum(A * A, axis=0) * np.sum(B * B, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den
        r[~np.isfinite(r)] = 0.0
    return r


def main():
    parser = argparse.ArgumentParser(description="Brain–language alignment with emotion features")
    parser.add_argument("--bids_root", type=str, required=True, help="Path to BIDS dataset root")
    parser.add_argument("--subject", type=str, required=True, help="BIDS subject ID, e.g., sub-001")
    parser.add_argument("--task", type=str, required=True, help="Task/story label, e.g., pieman")
    parser.add_argument("--run", type=str, default=None, help="Optional run label, e.g., 1")
    parser.add_argument("--align_json", type=str, default=None, help="Path to Gentle align.json for the story")
    parser.add_argument("--transcript", type=str, default=None, help="Plain-text transcript if no alignment JSON provided")
    parser.add_argument("--chunk_s", type=float, default=4.0, help="Chunk size in seconds for text analysis")
    parser.add_argument("--n_lags", type=int, default=4, help="Number of FIR lags (in TRs)")
    parser.add_argument("--alphas", type=str, default="1.0,10.0,100.0,1000.0", help="Ridge alphas (comma-separated)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for maps and metrics")
    parser.add_argument("--mock_emotion", action="store_true", help="Use lightweight heuristic emotion features (no transformers)")
    parser.add_argument("--save_design", action="store_true", help="Save design matrix (npz)")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load fMRI and events
    bold_fn = find_bold_file(args.bids_root, args.subject, args.task, args.run)
    nimg = nib.load(bold_fn)
    data = nimg.get_fdata()
    if data.ndim != 4:
        raise ValueError("Expected 4D fMRI data")
    n_trs = data.shape[-1]
    tr = float(nimg.header.get_zooms()[-1])

    events = load_events(args.bids_root, args.subject, args.task, args.run)
    # Heuristic: story onset is the first 'story' onset if present; else 0
    if "trial_type" in events.columns:
        story_rows = events[events["trial_type"].str.contains("story", na=False)]
    elif "event_type" in events.columns:
        story_rows = events[events["event_type"].str.contains("story", na=False)]
    else:
        story_rows = pd.DataFrame(columns=events.columns)
    onset_s = float(story_rows.iloc[0]["onset"]) if len(story_rows) else float(events.iloc[0]["onset"]) if "onset" in events.columns else 0.0
    duration_s = float(story_rows.iloc[0]["duration"]) if len(story_rows) and "duration" in story_rows.columns else (n_trs * tr - onset_s)

    # Alignment data
    if args.align_json and os.path.exists(args.align_json):
        words_df = load_alignment_gentle(args.align_json)
    else:
        if not args.transcript or not os.path.exists(args.transcript):
            raise FileNotFoundError("Provide --align_json or --transcript for text features")
        with open(args.transcript) as f:
            transcript = f.read()
        words_df = simple_uniform_alignment(transcript, duration_s)

    # Build chunks and compute emotion features
    chunks = build_text_chunks(words_df, chunk_s=args.chunk_s)
    if args.mock_emotion:
        emo_feats, emo_labels, feat_times = compute_emotion_features_mock(chunks)
    else:
        emo_pipe = load_emotion_pipeline()
        emo_feats, emo_labels, feat_times = compute_emotion_features(chunks, emo_pipe)

    # Resample to TR grid and build FIR design
    X = resample_features_to_tr(feat_times, emo_feats, tr=tr, n_trs=n_trs, start_offset_s=onset_s)
    X_fir = build_fir_design(X, n_lags=args.n_lags)
    if args.save_design:
        np.savez_compressed(os.path.join(args.out_dir, f"{args.subject}_task-{args.task}_design.npz"), X=X_fir, labels=np.array(emo_labels))

    # Prepare brain data
    mask = make_brain_mask(nimg)
    Y = data[mask].T  # [time, vox]

    # Temporal standardization
    X_fir = np.nan_to_num(X_fir, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    alphas = [float(a) for a in args.alphas.split(",")]
    Y_pred, _ = fit_ridge_encoding(X_fir, Y, alphas=alphas, n_splits=5)
    r = corrcoef_timewise(Y_pred, Y)

    # Save correlation map
    r_img = np.zeros(mask.shape, dtype=np.float32)
    r_img[mask] = r.astype(np.float32)
    out_img = nib.Nifti1Image(r_img, affine=nimg.affine, header=nimg.header)
    out_fn = os.path.join(args.out_dir, f"{args.subject}_task-{args.task}_emotion_alignment_r.nii.gz")
    nib.save(out_img, out_fn)

    # Also save summary JSON
    summary = {
        "subject": args.subject,
        "task": args.task,
        "run": args.run,
        "tr": tr,
        "n_trs": int(n_trs),
        "onset_s": onset_s,
        "duration_s": duration_s,
        "chunk_s": args.chunk_s,
        "n_lags": int(args.n_lags),
        "alphas": alphas,
        "emotion_labels": emo_labels,
        "mock_emotion": bool(args.mock_emotion),
        "output_map": out_fn
    }
    with open(os.path.join(args.out_dir, f"{args.subject}_task-{args.task}_emotion_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved correlation map to {out_fn}")


if __name__ == "__main__":
    main()

