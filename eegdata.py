import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "C:\\Users\\Soham\\research report project\\dataset\\OneDrive_2025-11-21\\PD REST"
OUTPUT_DIR = "C:\\Users\\Soham\\research report project\\dataset\\output"

# ----- Helper: robustly load EEGLAB-style .mat files -----
def load_eeglab_mat(path):
    """
    Loads a .mat file and tries to extract an EEGLAB 'EEG' structure.
    Returns: (data: np.ndarray channels x samples, fs: float, channels: list[str], meta: dict)
    """
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    # try common keys
    if "EEG" in mat:
        EEG = mat["EEG"]
        try:
            data = EEG.data
            fs = float(EEG.srate)
            # chanlocs -> labels
            channels = []
            if hasattr(EEG, "chanlocs") and EEG.chanlocs is not None:
                chanlocs = EEG.chanlocs
                # chanlocs may be array of structs
                try:
                    for ch in chanlocs:
                        if hasattr(ch, "labels"):
                            channels.append(str(ch.labels))
                        elif hasattr(ch, "label"):
                            channels.append(str(ch.label))
                except Exception:
                    # sometimes chanlocs nested differently
                    if hasattr(chanlocs, "labels"):
                        channels = [str(x) for x in chanlocs.labels]
            if len(channels) == 0:
                # fallback: generate generic names
                channels = [f"CH{i+1}" for i in range(data.shape[0])]
            # ensure shape channels x samples
            if data.shape[0] > data.shape[1]:
                # likely channels x time already; if not, we won't transpose blindly
                pass
            return data.astype(float), fs, channels, {"source": "EEG"}
        except Exception as e:
            raise RuntimeError(f"Could not parse EEG structure in {path}: {e}")
    else:
        # fallback: try to find any variable that looks like EEG data (2D numeric)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 1000:
                # heuristics: channel count between 1 and 256 and len > 500
                c, t = v.shape
                if 1 <= c <= 256 and t >= 256:
                    channels = [f"CH{i+1}" for i in range(c)]
                    fs = mat.get("srate", mat.get("fs", 250))  # guess
                    try:
                        fs = float(fs)
                    except:
                        fs = 250.0
                    return v.astype(float), fs, channels, {"source": k}
        raise RuntimeError(f"No EEG-like structure found in {path}.")

# ----- Filters -----
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs, lowcut=1.0, highcut=45.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.filtfilt(b, a, data, axis=1)

def apply_notch(data, fs, notch_freq=50.0, q=30.0):
    # notch filter (IIR)
    b, a = signal.iirnotch(notch_freq, q, fs)
    return signal.filtfilt(b, a, data, axis=1)

# ----- Preprocessing steps -----
def detrend(data):
    return signal.detrend(data, axis=1)

def avg_reference(data):
    # subtract average across channels
    mean = np.mean(data, axis=0, keepdims=True)
    return data - mean

def detect_bad_channels(data, z_thresh=3.0):
    # detect channels with extremely high variance relative to median variance
    vars = np.var(data, axis=1)
    med = np.median(vars)
    mad = np.median(np.abs(vars - med)) + 1e-9
    z = (vars - med) / mad
    bad_idx = np.where(np.abs(z) > z_thresh)[0]
    return list(bad_idx)

def simple_ica_remove(data, fs, n_components=None, var_thresh=0.8):
    """
    Fit ICA and remove components with extreme kurtosis or extremely high variance.
    Returns cleaned data and removed component indices.
    """
    X = data.T  # samples x channels for scikit-learn
    n_ch = data.shape[0]
    if n_components is None:
        n_components = min(n_ch, 20)
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    try:
        sources = ica.fit_transform(X)  # samples x comps
    except Exception:
        # fallback: skip ICA if fails
        return data, []
    comps_kurt = kurtosis(sources, axis=0)
    comps_var = np.var(sources, axis=0)
    # mark components to remove: large absolute kurtosis or huge variance components
    remove = np.where((np.abs(comps_kurt) > 10) | (comps_var > (np.median(comps_var) * 10)))[0].tolist()
    if len(remove) == 0:
        # conservative: keep as is
        return data, []
    # zero-out removed components and inverse transform
    sources[:, remove] = 0
    cleaned = ica.inverse_transform(sources)
    cleaned = cleaned.T
    return cleaned, remove

# ----- Feature extraction -----
def bandpower_welch(sig, fs, band, nperseg=1024):
    f, pxx = signal.welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapz(pxx[mask], f[mask]) if np.any(mask) else 0.0

def spectral_entropy(sig, fs, nperseg=1024):
    f, pxx = signal.welch(sig, fs=fs, nperseg=nperseg)
    ps = pxx / np.sum(pxx)
    return entropy(ps)

def hjorth_params(sig):
    # activity, mobility, complexity
    first_deriv = np.diff(sig)
    second_deriv = np.diff(sig, n=2)
    var0 = np.var(sig)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt((var2 / (var1 + 1e-12)) / (var1 / (var0 + 1e-12)) + 1e-12)
    return activity, mobility, complexity

def extract_features_from_record(data, fs, channels, filename):
    """
    data: channels x samples
    returns: pandas.DataFrame (1 row)
    """
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }
    feats = {}
    # per-channel features
    for i, ch in enumerate(channels):
        sig = data[i, :]
        feats[f"{ch}_mean"] = np.mean(sig)
        feats[f"{ch}_std"] = np.std(sig)
        feats[f"{ch}_skew"] = skew(sig)
        feats[f"{ch}_kurtosis"] = kurtosis(sig)
        # band powers
        total_power = 0.0
        band_pows = {}
        for bname, band in bands.items():
            p = bandpower_welch(sig, fs, band)
            band_pows[bname] = p
            feats[f"{ch}_p_{bname}"] = p
            total_power += p
        # relative power
        for bname in bands:
            feats[f"{ch}_rel_{bname}"] = (feats[f"{ch}_p_{bname}"] / (total_power + 1e-12))
        feats[f"{ch}_spec_ent"] = spectral_entropy(sig, fs)
        a, m, c = hjorth_params(sig)
        feats[f"{ch}_hj_act"] = a
        feats[f"{ch}_hj_mob"] = m
        feats[f"{ch}_hj_cmp"] = c

    feats["file"] = filename
    return pd.DataFrame([feats])

# ----- Main pipeline over all .mat files -----
def process_all_mat_files(data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                          lowcut=1.0, highcut=45.0, notch_freq=50.0,
                          do_ica=True, resample_fs=None):
    files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    rows = []
    for f in files:
        path = os.path.join(data_dir, f)
        print("Loading", path)
        try:
            data, fs, channels, meta = load_eeglab_mat(path)
        except Exception as e:
            print("SKIP (load error):", f, e)
            continue

        # ensure channels x samples
        if data.ndim == 1:
            print("SKIP (1D data):", f)
            continue
        if data.shape[0] > data.shape[1] and data.shape[0] < 200:
            # probably transposed (samples x channels) -> transpose
            data = data.T

        # optional resample
        if resample_fs is not None and resample_fs != fs:
            # resample to resample_fs
            n_samples = int(data.shape[1] * (resample_fs / fs))
            data = signal.resample(data, n_samples, axis=1)
            fs = resample_fs

        # detrend
        data = detrend(data)
        # notch
        data = apply_notch(data, fs, notch_freq=notch_freq)
        # bandpass
        data = apply_bandpass(data, fs, lowcut=lowcut, highcut=highcut, order=4)
        # re-reference to average
        data = avg_reference(data)

        # detect bad channels and drop them
        bad_idx = detect_bad_channels(data, z_thresh=5.0)
        if len(bad_idx) > 0:
            print(f"Detected bad channels in {f}: {bad_idx}")
            good_idx = [i for i in range(data.shape[0]) if i not in bad_idx]
            data = data[good_idx, :]
            channels = [channels[i] for i in good_idx]

        # ICA artifact removal
        removed_comps = []
        if do_ica:
            data, removed_comps = simple_ica_remove(data, fs, n_components=min(20, data.shape[0]))
            if len(removed_comps) > 0:
                print(f"Removed ICA comps for {f}: {removed_comps}")

        # save cleaned recording
        outfn = os.path.join(output_dir, f"cleaned_{os.path.splitext(f)[0]}.npz")
        np.savez(outfn, data=data, fs=fs, channels=np.array(channels), removed_comps=np.array(removed_comps))
        print("Saved cleaned:", outfn)

        # extract features
        df = extract_features_from_record(data, fs, channels, f)
        rows.append(df)

    if len(rows) == 0:
        print("No valid files processed.")
        return None

    features_df = pd.concat(rows, ignore_index=True).fillna(0)
    outcsv = os.path.join(output_dir, "eeg_features.csv")
    features_df.to_csv(outcsv, index=False)
    print("Wrote features CSV:", outcsv)
    return features_df

# ----- Example model training - baseline -----
def train_baseline_model(features_df, label_column="label"):
    """
    Example: we assume the user will add a 'label' column to the features_df
    describing class (0/1). If label column not present, we will not train.
    """
    if label_column not in features_df.columns:
        print(f"Label column '{label_column}' not found. Skipping model training.")
        return None

    X = features_df.drop(columns=[label_column, "file"])
    y = features_df[label_column].values

    # impute and scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    Xs = scaler.fit_transform(X_imp)

    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print("CV accuracy (5-fold):", np.round(scores, 4), "mean:", np.mean(scores))

    clf.fit(Xs, y)
    # save pipeline pieces
    joblib.dump({"model": clf, "imputer": imputer, "scaler": scaler}, os.path.join(OUTPUT_DIR, "rf_model.joblib"))
    print("Saved model to", os.path.join(OUTPUT_DIR, "rf_model.joblib"))

    # feature importances
    importances = clf.feature_importances_
    fi = pd.DataFrame({"feature": X.columns, "importance": importances})
    fi.sort_values("importance", ascending=False).to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
    print("Wrote feature importances.")
    return clf

# ----- Run pipeline -----
if __name__ == "__main__":
    print("Starting EEG pipeline... Looking for .mat files in", DATA_DIR)
    feat_df = process_all_mat_files(data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                                    lowcut=1.0, highcut=45.0, notch_freq=50.0,
                                    do_ica=True, resample_fs=None)

    # If you already have labels, merge them here
    # Example: create labels.csv with columns: file,label
    label_csv = os.path.join(OUTPUT_DIR, "labels.csv")
    if feat_df is not None and os.path.exists(label_csv):
        labels = pd.read_csv(label_csv)
        merged = feat_df.merge(labels, left_on="file", right_on="file", how="left")
        if "label" in merged.columns:
            train_baseline = train_baseline_model(merged, label_column="label")
        else:
            print("labels.csv found but 'label' column not present.")
    else:
        if feat_df is not None:
            print("No labels.csv found in /mnt/data. If you want to train, create labels.csv with columns 'file' and 'label' and re-run.")
    print("Pipeline finished.")