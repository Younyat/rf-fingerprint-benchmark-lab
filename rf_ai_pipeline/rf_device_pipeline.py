from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    metadata_glob: str
    data_suffix: str
    default_dtype: str
    model_kind: str


DATASETS: Dict[str, DatasetConfig] = {
    "kri_wifi": DatasetConfig(
        name="kri_wifi",
        root=ROOT / "neu_m044q5210" / "KRI-16Devices-RawData",
        metadata_glob="*.sigmf-meta",
        data_suffix=".sigmf-data",
        default_dtype="cf32",
        model_kind="random_forest",
    ),
    "uav_lightbridge": DatasetConfig(
        name="uav_lightbridge",
        root=ROOT / "neu_m046p309d" / "UAV-Sigmf-float16",
        metadata_glob="*.json",
        data_suffix=".bin",
        default_dtype="cf16_le",
        model_kind="extra_trees",
    ),
    "ieee_cbrs": DatasetConfig(
        name="ieee_cbrs",
        root=ROOT / "neu_4f197c103" / "IEEE_UPLOAD",
        metadata_glob="*.sigmf-meta",
        data_suffix=".sigmf-data",
        default_dtype="cf32_le",
        model_kind="hist_gradient_boosting",
    ),
    "wifi_dat_day": DatasetConfig(
        name="wifi_dat_day",
        root=ROOT / "WiFi-Dataset",
        metadata_glob="*.dat",
        data_suffix=".dat",
        default_dtype="cf32",
        model_kind="extra_trees",
    ),
}


MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "knn": {
        "family": "machine_learning_clasico",
        "paper_link": "KNN con features RF manuales, FD/AIB/SIB, PSD o bispectrum",
        "description": "Baseline de vecinos cercanos para comprobar separabilidad geometrica de las huellas RF.",
    },
    "svm_linear": {
        "family": "machine_learning_clasico",
        "paper_link": "SVM lineal como baseline de separabilidad",
        "description": "Modelo lineal fuerte para features normalizadas; util como control minimo.",
    },
    "svm_rbf": {
        "family": "machine_learning_clasico",
        "paper_link": "SVM RBF usado en UAV y features espectrales",
        "description": "Clasificador no lineal competitivo en datasets medianos con features tabulares.",
    },
    "random_forest": {
        "family": "machine_learning_clasico",
        "paper_link": "Random Forest con fingerprints manuales",
        "description": "Ensemble robusto para amplitud, fase, energia espectral y estadisticos IQ.",
    },
    "extra_trees": {
        "family": "machine_learning_clasico",
        "paper_link": "Ensemble de arboles como variante de Random Forest",
        "description": "Ensemble mas aleatorizado; suele funcionar bien con features heterogeneas.",
    },
    "logistic_regression": {
        "family": "baseline_lineal",
        "paper_link": "Regresion logistica comparada contra CNN en radio identification",
        "description": "Baseline probabilistico lineal para medir dificultad real del problema.",
    },
    "mlp": {
        "family": "red_neuronal_densa",
        "paper_link": "MLP/DNN sobre vectores PSD, CSP, waterfall rows o features",
        "description": "Red densa ligera; aqui opera sobre el vector de fingerprints extraido del IQ.",
    },
    "hist_gradient_boosting": {
        "family": "boosting",
        "paper_link": "Ensemble no lineal moderno para features tabulares",
        "description": "Boosting de histogramas; buen baseline no lineal sin dependencias deep learning.",
    },
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_meta(config: DatasetConfig, meta: dict) -> str:
    global_meta = meta.get("global") or meta.get("_metadata", {}).get("global") or {}
    return global_meta.get("core:datatype") or config.default_dtype


def sample_count_from_meta(meta: dict) -> Optional[int]:
    annotations = meta.get("annotations")
    if isinstance(annotations, dict):
        return annotations.get("core:sample_count")
    if isinstance(annotations, list) and annotations:
        return annotations[0].get("core:sample_count")
    nested = meta.get("_metadata", {}).get("annotations")
    if isinstance(nested, list) and nested:
        return nested[0].get("core:sample_count")
    return None


def data_path_for_meta(config: DatasetConfig, meta_path: Path) -> Path:
    if config.name == "uav_lightbridge":
        return meta_path.with_suffix(".bin")
    return meta_path.with_suffix(config.data_suffix)


def label_for_meta(config: DatasetConfig, meta_path: Path, meta: dict, ieee_target: str = "auto") -> str:
    if config.name == "wifi_dat_day":
        for parent in meta_path.parents:
            if re.fullmatch(r"Device_\d+", parent.name, flags=re.IGNORECASE):
                return parent.name
        match = re.search(r"tx_(\d+)", meta_path.name)
        if ieee_target == "tx":
            return f"tx_{match.group(1)}" if match else "unknown_tx"
        for parent in meta_path.parents:
            if re.fullmatch(r"Day_\d+", parent.name):
                return parent.name
        return f"tx_{match.group(1)}" if match else "unknown_dat"

    if config.name == "kri_wifi":
        match = re.search(r"WiFi_air_X310_([^_]+)_", meta_path.name)
        if not match:
            raise ValueError(f"No pude extraer device_id del nombre: {meta_path}")
        return match.group(1)

    if config.name == "uav_lightbridge":
        tx = (meta.get("annotations") or {}).get("transmitter") or {}
        return tx.get("core:device_id_genesys_lab") or tx.get("core:UAV") or "unknown_uav"

    if config.name == "ieee_cbrs":
        ann = (meta.get("annotations") or [{}])[0]
        rx = (ann.get("system_components:receiver") or {}).get("seid")
        tx = ann.get("system_components:transmitter") or {}
        tx_ids = []
        for value in tx.values():
            if isinstance(value, dict):
                seid = (value.get("signal:emitter") or {}).get("seid")
                if seid:
                    tx_ids.append(seid)
        label = ann.get("core:label")
        choices = {
            "tx_combo": "+".join(sorted(tx_ids)),
            "receiver": rx,
            "signal_label": label,
            "band": meta_path.parent.name,
        }
        if ieee_target != "auto":
            return choices.get(ieee_target) or "unknown_ieee"
        return choices["tx_combo"] or choices["receiver"] or choices["signal_label"] or choices["band"] or "unknown_ieee"

    raise KeyError(config.name)


def wifi_dat_manifest_records(config: DatasetConfig) -> Optional[List[dict]]:
    for manifest in [config.root / "labels.csv", config.root / "segments.csv"]:
        if not manifest.exists():
            continue
        records: List[dict] = []
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f), start=1):
                rel_path = row.get("data_path") or row.get("path") or row.get("file")
                label = row.get("label") or row.get("device_id") or row.get("device") or row.get("tx")
                if not rel_path or not label:
                    raise ValueError(
                        f"{manifest} fila {idx}: hacen falta columnas data_path/path/file y label/device_id/device/tx"
                    )
                data_path = (config.root / rel_path).resolve()
                if not data_path.exists():
                    data_path = (ROOT / rel_path).resolve()
                if not data_path.exists():
                    raise FileNotFoundError(f"{manifest} fila {idx}: no existe {rel_path}")
                data_size = data_path.stat().st_size
                dtype_name = row.get("dtype") or config.default_dtype
                total_samples = data_size // complex_sample_nbytes(dtype_name)
                start_sample = int(float(row.get("start_sample") or 0))
                end_raw = row.get("end_sample")
                end_sample = int(float(end_raw)) if end_raw not in (None, "") else total_samples
                if end_sample <= start_sample:
                    raise ValueError(f"{manifest} fila {idx}: end_sample debe ser mayor que start_sample")
                records.append(
                    {
                        "dataset": config.name,
                        "meta_path": data_path,
                        "data_path": data_path,
                        "label": label,
                        "dtype": dtype_name,
                        "sample_count": end_sample - start_sample,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "group": f"{data_path.relative_to(config.root)}:{start_sample}:{end_sample}",
                        "data_size_bytes": (end_sample - start_sample) * complex_sample_nbytes(dtype_name),
                    }
                )
        return records
    return None


def iter_records(
    config: DatasetConfig,
    limit_files: Optional[int] = None,
    max_files_per_class: Optional[int] = None,
    max_data_bytes: Optional[int] = None,
    ieee_target: str = "auto",
) -> Iterable[dict]:
    if config.name == "wifi_dat_day":
        manifest_records = wifi_dat_manifest_records(config)
        if manifest_records is not None:
            selected: List[dict] = []
            per_class: Dict[str, int] = {}
            selected_bytes = 0
            for record in manifest_records:
                label = record["label"]
                if max_files_per_class is not None and per_class.get(label, 0) >= max_files_per_class:
                    continue
                if max_data_bytes is not None and selected and selected_bytes + int(record["data_size_bytes"]) > max_data_bytes:
                    continue
                selected.append(record)
                selected_bytes += int(record["data_size_bytes"])
                per_class[label] = per_class.get(label, 0) + 1
                if limit_files and len(selected) >= limit_files:
                    break
            for record in selected:
                yield record
            return

    candidates: List[dict] = []
    for meta_path in sorted(config.root.rglob(config.metadata_glob)):
        if config.name == "wifi_dat_day":
            meta = {}
            data_path = meta_path
        else:
            meta = load_json(meta_path)
            data_path = data_path_for_meta(config, meta_path)
        if not data_path.exists():
            continue
        data_size = data_path.stat().st_size
        label = label_for_meta(config, meta_path, meta, ieee_target=ieee_target)
        candidates.append({
            "dataset": config.name,
            "meta_path": meta_path,
            "data_path": data_path,
            "label": label,
            "dtype": config.default_dtype if config.name == "wifi_dat_day" else dtype_from_meta(config, meta),
            "sample_count": data_size // complex_sample_nbytes(config.default_dtype) if config.name == "wifi_dat_day" else sample_count_from_meta(meta),
            "group": str(meta_path.relative_to(config.root)),
            "data_size_bytes": data_size,
        })

    selected: List[dict] = []
    per_class: Dict[str, int] = {}
    selected_bytes = 0
    labels = sorted({record["label"] for record in candidates})
    by_label = {label: [record for record in candidates if record["label"] == label] for label in labels}

    if max_data_bytes is not None and labels:
        min_balanced_bytes = sum(int(records[0]["data_size_bytes"]) for records in by_label.values() if records)
        if max_data_bytes < min_balanced_bytes:
            min_gb = min_balanced_bytes / (1024 ** 3)
            requested_gb = max_data_bytes / (1024 ** 3)
            raise ValueError(
                f"Presupuesto de datos demasiado pequeno para clasificacion multiclase: "
                f"{requested_gb:.3f} GB solo cubre parte de las clases. "
                f"Minimo balanceado aproximado para una captura por clase: {min_gb:.3f} GB."
            )
        progressed = True
        while progressed:
            progressed = False
            for label in labels:
                class_count = per_class.get(label, 0)
                class_records = by_label[label]
                if class_count >= len(class_records):
                    continue
                if max_files_per_class is not None and class_count >= max_files_per_class:
                    continue
                record = class_records[class_count]
                if selected and selected_bytes + int(record["data_size_bytes"]) > max_data_bytes:
                    continue
                selected.append(record)
                selected_bytes += int(record["data_size_bytes"])
                per_class[label] = class_count + 1
                progressed = True
                if limit_files and len(selected) >= limit_files:
                    progressed = False
                    break
    else:
        for record in candidates:
            label = record["label"]
            if max_files_per_class is not None and per_class.get(label, 0) >= max_files_per_class:
                continue
            selected.append(record)
            per_class[label] = per_class.get(label, 0) + 1
            if limit_files and len(selected) >= limit_files:
                break

    for record in selected:
        yield record


def read_iq_window(path: Path, dtype_name: str, start_sample: int, n_samples: int) -> np.ndarray:
    dtype_name = dtype_name.lower()
    if dtype_name in {"cf32", "cf32_le", "complex64"}:
        scalar_dtype = np.dtype("<f4")
    elif dtype_name in {"cf16_le", "float16"}:
        scalar_dtype = np.dtype("<f2")
    else:
        raise ValueError(f"Datatype IQ no soportado: {dtype_name}")

    byte_offset = start_sample * 2 * scalar_dtype.itemsize
    raw = np.fromfile(path, dtype=scalar_dtype, count=n_samples * 2, offset=byte_offset)
    if raw.size < 4:
        raise ValueError(f"Muy pocos datos IQ en {path}")
    raw = raw[: raw.size - (raw.size % 2)]
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def complex_sample_nbytes(dtype_name: str) -> int:
    dtype_name = dtype_name.lower()
    if dtype_name in {"cf32", "cf32_le", "complex64"}:
        return 8
    if dtype_name in {"cf16_le", "float16"}:
        return 4
    return 8


def feature_names(n_fft_bins: int = 12) -> List[str]:
    base = [
        "amp_mean",
        "amp_std",
        "amp_min",
        "amp_max",
        "amp_p10",
        "amp_p50",
        "amp_p90",
        "power_mean",
        "power_std",
        "real_mean",
        "real_std",
        "imag_mean",
        "imag_std",
        "phase_diff_mean",
        "phase_diff_std",
        "phase_diff_p10",
        "phase_diff_p90",
        "iq_corr",
        "papr",
        "zero_cross_real",
        "zero_cross_imag",
        "spectral_centroid",
        "spectral_spread",
        "spectral_flatness",
    ]
    return base + [f"fft_band_{i:02d}" for i in range(n_fft_bins)]


def extract_features(iq: np.ndarray, n_fft_bins: int = 12) -> np.ndarray:
    iq = iq[np.isfinite(iq.real) & np.isfinite(iq.imag)]
    if iq.size < 16:
        raise ValueError("Ventana IQ demasiado corta")
    iq = iq.astype(np.complex128, copy=False)
    iq = iq - np.mean(iq)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Ventana IQ con potencia invalida")
    iq = iq / rms
    amp = np.abs(iq).astype(np.float64)
    power = amp * amp
    real = iq.real
    imag = iq.imag
    phase = np.unwrap(np.angle(iq))
    phase_diff = np.diff(phase)
    real_std = float(np.std(real)) or 1e-12
    imag_std = float(np.std(imag)) or 1e-12
    iq_corr = float(np.mean((real - np.mean(real)) * (imag - np.mean(imag))) / (real_std * imag_std))
    papr = float(np.max(power) / (np.mean(power) + 1e-12))
    zcr_r = float(np.mean(np.diff(np.signbit(real)) != 0))
    zcr_i = float(np.mean(np.diff(np.signbit(imag)) != 0))

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq * np.hanning(iq.size)))) ** 2
    spectrum = spectrum + 1e-12
    freqs = np.linspace(-0.5, 0.5, spectrum.size, endpoint=False)
    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))
    flatness = float(math.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    bands = np.array_split(spectrum, n_fft_bins)
    band_energy = np.array([np.sum(b) / total for b in bands], dtype=np.float32)

    values = [
        float(np.mean(amp)),
        float(np.std(amp)),
        float(np.min(amp)),
        float(np.max(amp)),
        float(np.percentile(amp, 10)),
        float(np.percentile(amp, 50)),
        float(np.percentile(amp, 90)),
        float(np.mean(power)),
        float(np.std(power)),
        float(np.mean(real)),
        real_std,
        float(np.mean(imag)),
        imag_std,
        float(np.mean(phase_diff)),
        float(np.std(phase_diff)),
        float(np.percentile(phase_diff, 10)),
        float(np.percentile(phase_diff, 90)),
        iq_corr,
        papr,
        zcr_r,
        zcr_i,
        centroid,
        spread,
        flatness,
    ]
    return np.concatenate([np.asarray(values, dtype=np.float32), band_energy])


def windows_for_record(sample_count: Optional[int], window_size: int, windows_per_file: int) -> List[int]:
    if not sample_count or sample_count <= window_size:
        return [0]
    max_start = sample_count - window_size
    if windows_per_file <= 1:
        return [max_start // 2]
    return [int(x) for x in np.linspace(0, max_start, windows_per_file)]


def energy_ranked_windows(record: dict, window_size: int, windows_per_file: int, candidates: int = 48) -> List[int]:
    sample_count = record.get("sample_count")
    if not sample_count or sample_count <= window_size:
        return [0]
    max_start = sample_count - window_size
    starts = [int(x) for x in np.linspace(0, max_start, max(candidates, windows_per_file))]
    scored = []
    base_start = int(record.get("start_sample") or 0)
    for start in starts:
        try:
            iq = read_iq_window(record["data_path"], record["dtype"], base_start + start, window_size)
            power = float(np.mean(np.abs(iq.astype(np.complex128, copy=False)) ** 2))
            if np.isfinite(power):
                scored.append((power, start))
        except Exception:
            continue
    if not scored:
        return windows_for_record(sample_count, window_size, windows_per_file)
    scored.sort(reverse=True)
    selected = sorted({start for _, start in scored[:windows_per_file]})
    return selected or windows_for_record(sample_count, window_size, windows_per_file)


def select_windows_for_record(record: dict, window_size: int, windows_per_file: int, strategy: str) -> List[int]:
    if strategy == "energy":
        return energy_ranked_windows(record, window_size, windows_per_file)
    return windows_for_record(record.get("sample_count"), window_size, windows_per_file)


def build_matrix(
    config: DatasetConfig,
    limit_files: Optional[int],
    max_files_per_class: Optional[int],
    max_data_bytes: Optional[int],
    window_size: int,
    windows_per_file: int,
    window_strategy: str = "linspace",
    ieee_target: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    rows: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []
    used: List[dict] = []
    for record in iter_records(
        config,
        limit_files=limit_files,
        max_files_per_class=max_files_per_class,
        max_data_bytes=max_data_bytes,
        ieee_target=ieee_target,
    ):
        starts = select_windows_for_record(record, window_size, windows_per_file, window_strategy)
        ok_windows = 0
        for start in starts:
            try:
                iq = read_iq_window(
                    record["data_path"],
                    record["dtype"],
                    int(record.get("start_sample") or 0) + start,
                    window_size,
                )
                rows.append(extract_features(iq))
                labels.append(record["label"])
                groups.append(record["group"])
                ok_windows += 1
            except Exception as exc:
                record["last_error"] = str(exc)
        if ok_windows:
            used.append({**record, "windows": ok_windows})
    if not rows:
        raise RuntimeError(f"No se pudieron extraer features para {config.name}")
    return np.vstack(rows), np.asarray(labels), np.asarray(groups), used


def make_estimator(kind: str, random_state: int) -> Pipeline:
    if kind not in MODEL_CATALOG:
        raise ValueError(f"Modelo no soportado: {kind}. Opciones: {', '.join(MODEL_CATALOG)}")
    if kind == "knn":
        return Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))])
    if kind == "svm_linear":
        clf = SVC(kernel="linear", class_weight="balanced", probability=False, random_state=random_state)
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    if kind == "svm_rbf":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=False, random_state=random_state)
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    if kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    if kind == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    if kind == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="auto",
            random_state=random_state,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    if kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=260,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=160,
        l2_regularization=0.05,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def stratified_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_groups = set()
    test_groups = set()
    for label in sorted(set(labels.tolist())):
        label_groups = sorted(set(groups[labels == label].tolist()))
        if len(label_groups) < 2:
            train_groups.update(label_groups)
            continue
        rng.shuffle(label_groups)
        n_test = max(1, int(round(len(label_groups) * test_size)))
        n_test = min(n_test, len(label_groups) - 1)
        test_groups.update(label_groups[:n_test])
        train_groups.update(label_groups[n_test:])
    train_idx = np.asarray([idx for idx, group in enumerate(groups) if group in train_groups], dtype=int)
    test_idx = np.asarray([idx for idx, group in enumerate(groups) if group in test_groups], dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return train_idx, test_idx


def write_inventory(config: DatasetConfig, args: argparse.Namespace) -> Path:
    records = list(
        iter_records(
            config,
            limit_files=args.limit_files,
            max_files_per_class=args.max_files_per_class,
            max_data_bytes=getattr(args, "max_data_bytes", None),
            ieee_target=args.ieee_target,
        )
    )
    counts: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    examples = []
    for rec in records:
        counts[rec["label"]] = counts.get(rec["label"], 0) + 1
        dtypes[rec["dtype"]] = dtypes.get(rec["dtype"], 0) + 1
        if len(examples) < 5:
            examples.append(
                {
                    "meta": str(rec["meta_path"].relative_to(ROOT)),
                    "data": str(rec["data_path"].relative_to(ROOT)),
                    "label": rec["label"],
                    "dtype": rec["dtype"],
                    "sample_count": rec["sample_count"],
                    "data_size_bytes": rec.get("data_size_bytes"),
                    "start_sample": rec.get("start_sample"),
                    "end_sample": rec.get("end_sample"),
                }
            )
    payload = {
        "dataset": config.name,
        "root": str(config.root.relative_to(ROOT)),
        "records": len(records),
        "data_size_bytes": sum(int(r.get("data_size_bytes") or 0) for r in records),
        "data_size_gb": sum(int(r.get("data_size_bytes") or 0) for r in records) / (1024 ** 3),
        "classes": counts,
        "dtypes": dtypes,
        "examples": examples,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    has_subset_filter = bool(
        args.limit_files is not None
        or args.max_files_per_class is not None
        or getattr(args, "max_data_bytes", None) is not None
    )
    suffix = "_inventory_subset.json" if has_subset_filter else "_inventory.json"
    out = REPORTS_DIR / f"{config.name}{suffix}"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def train_dataset(config: DatasetConfig, args: argparse.Namespace) -> Path:
    model_kind = getattr(args, "model_kind", None) or config.model_kind
    x, labels, groups, used = build_matrix(
        config,
        limit_files=args.limit_files,
        max_files_per_class=args.max_files_per_class,
        max_data_bytes=getattr(args, "max_data_bytes", None),
        window_size=args.window_size,
        windows_per_file=args.windows_per_file,
        window_strategy=getattr(args, "window_strategy", "linspace"),
        ieee_target=args.ieee_target,
    )
    class_names = sorted(set(labels.tolist()))
    if len(class_names) < 2:
        raise RuntimeError(
            f"{config.name} solo tiene una clase ({class_names}). "
            "No se puede entrenar un identificador supervisado con una sola identidad."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    if config.name == "wifi_dat_day":
        all_idx = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels,
        )
        split_protocol = (
            "Split estratificado por ventanas dentro de cada archivo .dat. "
            "Validacion debil: las ventanas de la misma captura pueden aparecer en train y test porque solo hay un archivo por clase."
        )
    else:
        train_idx, test_idx = stratified_group_split(labels, groups, args.test_size, args.random_state)
        split_protocol = "Split estratificado por clase y separado por archivo/captura; ninguna captura usada como grupo aparece a la vez en train y test."
    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    holdout_used = [r for r in used if r["group"] in test_groups]
    train_used = [r for r in used if r["group"] in train_groups]
    model = make_estimator(model_kind, args.random_state)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    acc = accuracy_score(y[test_idx], pred)
    balanced_acc = balanced_accuracy_score(y[test_idx], pred)
    macro_f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
    referenced_bytes = sum(int(r.get("data_size_bytes") or 0) for r in used)
    selection_reads_per_record = 48 if getattr(args, "window_strategy", "linspace") == "energy" else 0
    estimated_iq_bytes_read = sum(
        int(r.get("windows") or 0) * int(args.window_size) * complex_sample_nbytes(str(r.get("dtype") or config.default_dtype))
        + selection_reads_per_record * int(args.window_size) * complex_sample_nbytes(str(r.get("dtype") or config.default_dtype))
        for r in used
    )

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    bundle = {
        "dataset": config.name,
        "model_kind": model_kind,
        "model_metadata": MODEL_CATALOG[model_kind],
        "model": model,
        "label_encoder": encoder,
        "feature_names": feature_names(),
        "window_size": args.window_size,
        "windows_per_file": args.windows_per_file,
        "window_strategy": getattr(args, "window_strategy", "linspace"),
        "ieee_target": args.ieee_target,
        "dtype_default": config.default_dtype,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "classes": encoder.classes_.tolist(),
        "source_records": [
            {
                "meta_path": str(r["meta_path"].relative_to(ROOT)),
                "data_path": str(r["data_path"].relative_to(ROOT)),
                "label": r["label"],
                "dtype": r["dtype"],
                "sample_count": r["sample_count"],
                "data_size_bytes": r.get("data_size_bytes"),
                "start_sample": r.get("start_sample"),
                "end_sample": r.get("end_sample"),
                "windows": r["windows"],
            }
            for r in used
        ],
        "train_record_groups": sorted(train_groups),
        "holdout_record_groups": sorted(test_groups),
    }
    model_path = MODELS_DIR / f"{config.name}_{model_kind}_model.joblib"
    joblib.dump(bundle, model_path)
    model_size_bytes = model_path.stat().st_size

    report = {
        "dataset": config.name,
        "model_kind": model_kind,
        "model_metadata": MODEL_CATALOG[model_kind],
        "model_path": str(model_path.relative_to(ROOT)),
        "model_size_bytes": model_size_bytes,
        "records_used": len(used),
        "referenced_data_size_bytes": referenced_bytes,
        "data_size_bytes": referenced_bytes,
        "estimated_iq_bytes_read": estimated_iq_bytes_read,
        "estimated_iq_gb_read": estimated_iq_bytes_read / (1024 ** 3),
        "requested_max_data_gb": getattr(args, "max_data_gb", None),
        "requested_max_data_percent": getattr(args, "max_data_percent", None),
        "requested_max_data_bytes": getattr(args, "max_data_bytes", None),
        "window_strategy": getattr(args, "window_strategy", "linspace"),
        "windows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "classes": encoder.classes_.tolist(),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "train_records": len(train_used),
        "holdout_records_count": len(holdout_used),
        "split_protocol": split_protocol,
        "holdout_records": [
            {
                "meta_path": str(r["meta_path"].relative_to(ROOT)),
                "data_path": str(r["data_path"].relative_to(ROOT)),
                "label": r["label"],
                "dtype": r["dtype"],
                "sample_count": r["sample_count"],
                "data_size_bytes": r.get("data_size_bytes"),
                "start_sample": r.get("start_sample"),
                "end_sample": r.get("end_sample"),
                "windows": r["windows"],
                "group": r["group"],
            }
            for r in holdout_used[:100]
        ],
        "holdout_accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "classification_report": classification_report(
            y[test_idx],
            pred,
            target_names=encoder.classes_,
            labels=list(range(len(encoder.classes_))),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y[test_idx], pred).tolist(),
        "sample_predictions": prediction_samples(bundle, holdout_used[: min(5, len(holdout_used))]),
    }
    report_path = REPORTS_DIR / f"{config.name}_{model_kind}_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path = REPORTS_DIR / f"{config.name}_validation.json"
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_model = MODELS_DIR / f"{config.name}_device_model.joblib"
    joblib.dump(bundle, latest_model)
    print(json.dumps({"model": str(model_path), "report": str(report_path), "accuracy": acc}, indent=2))
    return model_path


def prediction_samples(bundle: dict, records: Sequence[dict]) -> List[dict]:
    samples = []
    for rec in records:
        pred = predict_record(
            bundle,
            rec["data_path"],
            rec["dtype"],
            rec.get("sample_count"),
            top_k=3,
            start_sample=int(rec.get("start_sample") or 0),
        )
        samples.append(
            {
                "file": str(rec["data_path"].relative_to(ROOT)),
                "expected": rec["label"],
                "predicted": pred["prediction"],
                "confidence": pred["confidence"],
                "top_k": pred["top_k"],
            }
        )
    return samples


def predict_record(
    bundle: dict,
    data_path: Path,
    dtype_name: str,
    sample_count: Optional[int],
    top_k: int = 5,
    start_sample: int = 0,
) -> dict:
    pseudo_record = {"data_path": data_path, "dtype": dtype_name, "sample_count": sample_count}
    starts = select_windows_for_record(
        pseudo_record,
        int(bundle["window_size"]),
        3,
        str(bundle.get("window_strategy") or "linspace"),
    )
    feats = []
    for start in starts:
        iq = read_iq_window(data_path, dtype_name, int(start_sample or 0) + start, int(bundle["window_size"]))
        feats.append(extract_features(iq))
    x = np.vstack(feats)
    model: Pipeline = bundle["model"]
    encoder: LabelEncoder = bundle["label_encoder"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x).mean(axis=0)
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        top = [
            {"label": str(encoder.inverse_transform([int(i)])[0]), "probability": float(proba[int(i)])}
            for i in order[:top_k]
        ]
        return {"prediction": top[0]["label"], "confidence": top[0]["probability"], "top_k": top}
    pred = model.predict(x)
    values, counts = np.unique(pred, return_counts=True)
    best = int(values[np.argmax(counts)])
    return {
        "prediction": str(encoder.inverse_transform([best])[0]),
        "confidence": float(np.max(counts) / np.sum(counts)),
        "top_k": [],
    }


def predict_cli(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    dataset = bundle["dataset"]
    config = DATASETS[dataset]
    meta_path = Path(args.meta)
    meta = load_json(meta_path)
    data_path = Path(args.data) if args.data else data_path_for_meta(config, meta_path)
    dtype_name = args.dtype or dtype_from_meta(config, meta)
    sample_count = sample_count_from_meta(meta)
    expected = None
    try:
        expected = label_for_meta(config, meta_path, meta, ieee_target=bundle.get("ieee_target", "auto"))
    except Exception:
        pass
    pred = predict_record(bundle, data_path, dtype_name, sample_count, top_k=args.top_k)
    pred["meta"] = str(meta_path)
    pred["data"] = str(data_path)
    pred["expected_from_metadata"] = expected
    print(json.dumps(pred, indent=2))


def write_csv_summary() -> Path:
    rows = []
    for report_path in sorted(REPORTS_DIR.glob("*_validation.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "model_kind" not in report:
            continue
        if report_path.name == f"{report['dataset']}_validation.json":
            continue
        rows.append(
            {
                "dataset": report["dataset"],
                "model_kind": report.get("model_kind", "unknown"),
                "model_path": report["model_path"],
                "records_used": report["records_used"],
                "windows": report["windows"],
                "classes": len(report["classes"]),
                "holdout_accuracy": report["holdout_accuracy"],
                "balanced_accuracy": report.get("balanced_accuracy"),
                "macro_f1": report.get("macro_f1"),
            }
        )
    out = REPORTS_DIR / "models_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model_kind",
                "model_path",
                "records_used",
                "windows",
                "classes",
                "holdout_accuracy",
                "balanced_accuracy",
                "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out


def resolve_dataset(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise SystemExit(f"Dataset no soportado: {name}. Usa uno de: {', '.join(DATASETS)}")
    return DATASETS[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="RF device fingerprinting pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    common.add_argument("--limit-files", type=int, default=None)
    common.add_argument("--max-files-per-class", type=int, default=None)
    common.add_argument("--max-data-gb", type=float, default=None)
    common.add_argument("--ieee-target", choices=["auto", "tx_combo", "receiver", "signal_label", "band"], default="auto")
    common.add_argument("--model-kind", choices=sorted(MODEL_CATALOG), default=None)
    common.add_argument("--window-strategy", choices=["linspace", "energy"], default="linspace")

    p_discover = sub.add_parser("discover", parents=[common])
    p_discover.set_defaults(func=lambda a: print(write_inventory(resolve_dataset(a.dataset), a)))

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--window-size", type=int, default=4096)
    p_train.add_argument("--windows-per-file", type=int, default=3)
    p_train.add_argument("--test-size", type=float, default=0.25)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.set_defaults(func=lambda a: train_dataset(resolve_dataset(a.dataset), a))

    p_retrain = sub.add_parser("retrain", parents=[common])
    p_retrain.add_argument("--window-size", type=int, default=4096)
    p_retrain.add_argument("--windows-per-file", type=int, default=3)
    p_retrain.add_argument("--test-size", type=float, default=0.25)
    p_retrain.add_argument("--random-state", type=int, default=42)
    p_retrain.set_defaults(func=lambda a: train_dataset(resolve_dataset(a.dataset), a))

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--model", required=True)
    p_predict.add_argument("--meta", required=True)
    p_predict.add_argument("--data", default=None)
    p_predict.add_argument("--dtype", default=None)
    p_predict.add_argument("--top-k", type=int, default=5)
    p_predict.set_defaults(func=predict_cli)

    p_summary = sub.add_parser("summary")
    p_summary.set_defaults(func=lambda a: print(write_csv_summary()))

    args = parser.parse_args()
    if getattr(args, "max_data_gb", None) is not None:
        args.max_data_bytes = int(float(args.max_data_gb) * 1024 * 1024 * 1024)
    else:
        args.max_data_bytes = None
    args.func(args)


if __name__ == "__main__":
    main()
