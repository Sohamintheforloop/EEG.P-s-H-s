# robust_eeg_pipeline.py
import os, re, glob, math, json
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
OUTPUT_DIR = "D:\\research report project\\dataset\\output"
SR_FALLBACK = 250
EPOCH_SEC = 2.0
EPOCH_OVERLAP = 0.5
BAND_RANGES = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta":  (13,30),
    "gamma": (30,45)
}
RANDOM_STATE = 42
N_JOBS = -1
MODEL_OUT = "trained_rf_model.joblib"
FEATURES_OUT = "features_table.csv"
RESULTS_OUT = "cv_results.json"
# ----------------------------

def load_npz_try(path):
    try:
        data = np.load(path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return {}

def inspect_npz(path):
    d = load_npz_try(path)
    print(f"\nFILE: {os.path.basename(path)}")
    if not d:
        print("  - No readable arrays or failed to load.")
        return d
    for k, v in d.items():
        t = type(v)
        if isinstance(v, np.ndarray):
            print(f"  - {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  - {k}: {t}")
    return d

def infer_data_array(dct, filename=None):
    """
    More robust detection:
    - If 3D ndarray: interpret as epoched data if dims match (epochs, channels, samples) or (epochs, samples, channels)
    - If 2D ndarray: channels x samples or samples x channels (heuristic)
    - If 1D ndarray: single channel continuous
    """
    data_keys = ['data','eeg','signals','X','raw','signal','cleaned','ica','eeg_data','eegClean']
    sr_keys = ['sr','fs','srate','sampling_rate']
    label_keys = ['label','y','target','class']
    subject_keys = ['subject','sub','sid','id']

    eeg = None; sr = None; label = None; subject = None; epoched = False

    # Look for common keys first
    for k in data_keys:
        if k in dct:
            eeg = dct[k]
            break

    # if nothing, pick the first ndarray (prefer 3D then 2D)
    if eeg is None:
        nds = [(k,v) for k,v in dct.items() if isinstance(v, np.ndarray)]
        # prefer 3D arrays
        nds_3 = [item for item in nds if item[1].ndim == 3]
        nds_2 = [item for item in nds if item[1].ndim == 2]
        nds_1 = [item for item in nds if item[1].ndim == 1]
        if nds_3:
            eeg = nds_3[0][1]
        elif nds_2:
            eeg = nds_2[0][1]
        elif nds_1:
            eeg = nds_1[0][1]

    # sampling rate
    for k in sr_keys:
        if k in dct:
            try:
                sr = float(np.asarray(dct[k]).ravel()[0])
            except:
                sr = None
            break

    # label
    for k in label_keys:
        if k in dct:
            try:
                v = np.asarray(dct[k])
                if v.size == 1:
                    label = str(v.ravel()[0])
                else:
                    label = v
            except:
                label = str(dct[k])
            break

    # subject id
    for k in subject_keys:
        if k in dct:
            try:
                subject = str(np.asarray(dct[k]).ravel()[0])
            except:
                subject = str(dct[k])
            break

    # If nothing found, try filename inference
    if filename is not None and (label is None or subject is None):
        f_label, f_sub = try_infer_from_filename(filename)
        if label is None:
            label = f_label
        if subject is None:
            subject = f_sub

    # Interpret eeg shape and return standardized representation
    if eeg is None:
        return None, sr, label, subject, False

    eeg = np.asarray(eeg)
    if eeg.ndim == 3:
        # assume (epochs, channels, samples) or (epochs, samples, channels)
        # try to detect which: pick the dimension with largest length -> samples
        dims = eeg.shape
        # if middle dim is small (<100) treat as channels
        if dims[1] < dims[2]:
            # likely (epochs, channels, samples)
            ep = eeg
            epoched = True
            # standardize to (epochs, channels, samples)
            return ep, sr, label, subject, epoched
        else:
            # maybe (epochs, samples, channels) -> transpose last two
            ep = eeg.transpose(0,2,1)
            epoched = True
            return ep, sr, label, subject, epoched
    elif eeg.ndim == 2:
        # could be (channels, samples) or (samples, channels)
        ch, sm = eeg.shape
        # Heuristic: if channels < samples then assume (channels, samples)
        if ch < sm:
            return eeg, sr, label, subject, False
        else:
            # transpose
            return eeg.T, sr, label, subject, False
    elif eeg.ndim == 1:
        # single channel continuous
        return eeg[np.newaxis, :], sr, label, subject, False
    else:
        return None, sr, label, subject, False

def try_infer_from_filename(fname):
    base = os.path.basename(fname).lower()
    label = None; subject = None
    if 'pd' in base or 'parkin' in base:
        label = 'PD'
    elif 'ctrl' in base or 'control' in base or 'healthy' in base or 'hc' in base:
        label = 'control'
    m = re.search(r'(sub|subject|sid|id)?[_\-]?(\d{2,4})', base)
    if m:
        subject = m.group(2)
    else:
        m2 = re.search(r'(\d+)', base)
        if m2:
            subject = m2.group(1)
    return label, subject

def sliding_epochs_from_array(arr_channels_samples, sr, epoch_sec=2.0, overlap=0.5):
    """
    Input: arr_channels_samples -> shape (n_channels, n_samples)
    Returns: list of epochs where each epoch is shape (n_channels, epoch_samples)
    """
    n_ch, n_samp = arr_channels_samples.shape
    epoch_len = int(epoch_sec * sr)
    step = int(epoch_len * (1 - overlap))
    if step <= 0:
        raise ValueError("EPOCH_OVERLAP too large, step <=0")
    epochs = []
    for start in range(0, n_samp - epoch_len + 1, step):
        ep = arr_channels_samples[:, start:start+epoch_len]
        epochs.append(ep)
    return epochs

def bandpower_from_psd(f, Pxx, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

def extract_features_from_epoch_multichannel(epoch_array, sr):
    """
    epoch_array: shape (n_channels, epoch_samples)
    Return flattened features averaged across channels (prefixed by ch index originally).
    """
    n_ch, _ = epoch_array.shape
    # compute per-channel features and then average
    per_ch = []
    for ch in range(n_ch):
        epoch = epoch_array[ch]
        feats = {}
        feats['mean'] = np.mean(epoch)
        feats['std'] = np.std(epoch)
        feats['var'] = np.var(epoch)
        feats['skew'] = skew(epoch)
        feats['kurtosis'] = kurtosis(epoch)
        pxx_hist, _ = np.histogram(epoch, bins=50, density=True)
        pxx_hist = pxx_hist[pxx_hist>0]
        feats['entropy'] = entropy(pxx_hist) if pxx_hist.size>0 else 0.0
        f, Pxx = signal.welch(epoch, fs=sr, nperseg=min(256, len(epoch)))
        total_power = np.trapz(Pxx, f) + 1e-12
        for band, (lo,hi) in BAND_RANGES.items():
            bp = bandpower_from_psd(f, Pxx, lo, hi)
            feats[f'bandpower_{band}'] = bp
            feats[f'relband_{band}'] = bp / total_power
        feats['spec_centroid'] = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12)
        per_ch.append(feats)
    # average across channels
    keys = per_ch[0].keys()
    combined = {k: float(np.mean([pc[k] for pc in per_ch])) for k in keys}
    return combined

def process_file(path, epoch_sec=EPOCH_SEC, overlap=EPOCH_OVERLAP, verbose=False):
    d = load_npz_try(path)
    if verbose:
        inspect_npz(path)
    eeg_arr, sr, label, subject, epoched = infer_data_array(d, filename=path)
    f_label, f_sub = try_infer_from_filename(path)
    if label is None:
        label = f_label
    if subject is None:
        subject = f_sub or os.path.splitext(os.path.basename(path))[0]

    if sr is None:
        sr = SR_FALLBACK

    if eeg_arr is None:
        raise RuntimeError(f"No suitable EEG array found in {os.path.basename(path)}")

    rows = []
    # Case 1: epoched already -> eeg_arr shape (n_epochs, n_channels, n_samples)
    if epoched:
        ep_arr = np.asarray(eeg_arr)
        n_epochs = ep_arr.shape[0]
        for ep_idx in range(n_epochs):
            epoch = ep_arr[ep_idx]  # shape (channels, samples) expected
            # fix if order is (channels, samples) or (samples, channels)
            if epoch.ndim == 2:
                if epoch.shape[0] < epoch.shape[1]:
                    pass  # (channels, samples)
                else:
                    epoch = epoch.T
            elif epoch.ndim == 1:
                epoch = epoch[np.newaxis, :]
            feats = extract_features_from_epoch_multichannel(epoch, sr)
            # prefix keys with channel-agg marker
            feats = {f: v for f, v in feats.items()}
            feats.update({
                'file': os.path.basename(path),
                'filepath': os.path.abspath(path),
                'epoch_idx': int(ep_idx),
                'n_channels': int(epoch.shape[0]),
                'n_samples': int(epoch.shape[1]),
                'sr': float(sr),
                'epoch_sec': float(epoch.shape[1]/sr),
                'label': str(label) if label is not None else None,
                'subject': str(subject)
            })
            rows.append(feats)
    else:
        # Continuous 2D array: (n_channels, n_samples)
        arr = np.asarray(eeg_arr)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected arr shape for {path}: {arr.shape}")
        n_ch, n_samp = arr.shape
        # if too short for a single epoch -> skip
        epoch_len = int(epoch_sec * sr)
        if n_samp < epoch_len:
            raise RuntimeError(f"File {os.path.basename(path)} too short for one epoch: n_samples={n_samp}, epoch_len={epoch_len}")
        epochs = sliding_epochs_from_array(arr, sr, epoch_sec=epoch_sec, overlap=overlap)
        for ep_idx, ep in enumerate(epochs):
            feats = extract_features_from_epoch_multichannel(ep, sr)
            feats.update({
                'file': os.path.basename(path),
                'filepath': os.path.abspath(path),
                'epoch_idx': int(ep_idx),
                'n_channels': int(ep.shape[0]),
                'n_samples': int(ep.shape[1]),
                'sr': float(sr),
                'epoch_sec': float(ep.shape[1]/sr),
                'label': str(label) if label is not None else None,
                'subject': str(subject)
            })
            rows.append(feats)
    df = pd.DataFrame(rows)
    return df

def build_feature_table(npz_folder=OUTPUT_DIR, pattern="*.npz", verbose=True):
    files = sorted(glob.glob(os.path.join(npz_folder, pattern)))
    dfs = []
    skipped = []
    if verbose:
        print(f"Found {len(files)} .npz files in {npz_folder}")
    for f in files:
        try:
            df = process_file(f, epoch_sec=EPOCH_SEC, overlap=EPOCH_OVERLAP, verbose=verbose)
            if df is not None and len(df)>0:
                dfs.append(df)
            else:
                skipped.append((f, "No epochs returned"))
        except Exception as e:
            skipped.append((f, str(e)))
            if verbose:
                print(f"Skipping {os.path.basename(f)}: {e}")
    if len(dfs) == 0:
        # give a helpful summary and raise
        print("\nNo feature DataFrames created. Summary of files checked:")
        for f, reason in skipped:
            print(f" - {os.path.basename(f)} : {reason}")
        raise RuntimeError("No epochs/features extracted. See summary above — filenames and reasons were printed.")
    features_df = pd.concat(dfs, ignore_index=True)
    # ensure label exists; if missing try to infer from filename tokens
    if 'label' not in features_df.columns:
        features_df['label'] = features_df['file'].apply(lambda x: ('PD' if 'pd' in x.lower() or 'parkin' in x.lower() else ('control' if 'ctrl' in x.lower() or 'control' in x.lower() or 'hc' in x.lower() else None)))
    if features_df['label'].isnull().any():
        n_missing = features_df['label'].isnull().sum()
        print(f"Warning: {n_missing} epochs have missing label. Dropping them.")
        features_df = features_df[features_df['label'].notnull()].copy()
    return features_df

# The rest of your training pipeline remains the same (copy from your original functions)
def train_and_evaluate(features_df, group_col='subject', label_col='label'):
    non_feat = ['file','filepath','epoch_idx','n_channels','n_samples','sr','epoch_sec', label_col, group_col]
    feat_cols = [c for c in features_df.columns if c not in non_feat]
    X = features_df[feat_cols].values
    y = features_df[label_col].values
    groups = features_df[group_col].values if group_col in features_df else None

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    print("Classes:", classes)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
    ])

    if groups is not None and len(np.unique(groups)) > 1:
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        cv = gkf
        cv_kwargs = {'groups': groups}
        print(f"Using GroupKFold with {min(5, len(np.unique(groups)))} splits based on {group_col}.")
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv = skf
        cv_kwargs = {}
        print("Using StratifiedKFold with 5 splits.")

    scores = cross_val_score(pipe, X, y_enc, cv=cv, **cv_kwargs, scoring='balanced_accuracy', n_jobs=N_JOBS)
    print("Baseline balanced accuracy (CV):", np.round(scores,4))
    print("Baseline mean:", float(np.mean(scores)))

    param_dist = {
        'clf__n_estimators': [100,200,400,800],
        'clf__max_depth': [None, 10, 20, 40, 80],
        'clf__min_samples_split': [2,5,10],
        'clf__min_samples_leaf': [1,2,4],
        'clf__max_features': ['sqrt','log2', None]
    }

    rs = RandomizedSearchCV(pipe, param_dist, n_iter=30, cv=cv, random_state=RANDOM_STATE, n_jobs=N_JOBS, scoring='balanced_accuracy', verbose=1)
    rs.fit(X, y_enc, **({'groups': groups} if groups is not None and len(np.unique(groups))>1 else {}))

    print("Best CV score:", rs.best_score_)
    print("Best params:", rs.best_params_)

    best = rs.best_estimator_
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(best, X, y_enc, cv=cv, **cv_kwargs, n_jobs=N_JOBS)
    bacc = balanced_accuracy_score(y_enc, y_pred)
    cm = confusion_matrix(y_enc, y_pred)
    report = classification_report(y_enc, y_pred, target_names=le.classes_, output_dict=True)
    print("Aggregated Balanced Accuracy:", bacc)
    print("Confusion Matrix:\n", cm)
    print("Classification report:\n", classification_report(y_enc, y_pred, target_names=le.classes_))

    results = {
        'label_encoder_classes': classes,
        'best_cv_score': float(rs.best_score_),
        'best_params': rs.best_params_,
        'agg_balanced_accuracy': float(bacc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    best.fit(X, y_enc)
    return best, feat_cols, le, results

if __name__ == "__main__":
    features_df = build_feature_table(npz_folder=OUTPUT_DIR, pattern="*.npz", verbose=True)
    print("Feature table shape:", features_df.shape)
    features_df.to_csv(FEATURES_OUT, index=False)
    if features_df['label'].nunique() < 2:
        raise RuntimeError("Need at least two classes to train. Found labels: {}".format(features_df['label'].unique()))
    model, feature_cols, label_encoder, results = train_and_evaluate(features_df)
    print("Saving model to", MODEL_OUT)
    joblib.dump({
        'model': model,
        'feature_columns': feature_cols,
        'label_encoder': label_encoder
    }, MODEL_OUT)
    with open(RESULTS_OUT, 'w') as fh:
        json.dump(results, fh, indent=2)
    print("Done. Artifacts written:")
    print(" -", FEATURES_OUT)
    print(" -", MODEL_OUT)
    print(" -", RESULTS_OUT)
