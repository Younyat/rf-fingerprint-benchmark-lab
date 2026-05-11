from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import joblib

import rf_device_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "web"
MODEL_SERIES_ORDER = [
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "mlp",
    "svm_rbf",
    "knn",
    "svm_linear",
    "logistic_regression",
]


@dataclass
class Job:
    id: str
    operation: str
    status: str
    created_at: float
    updated_at: float
    params: Dict[str, Any]
    progress_percent: float = 0.0
    progress_label: str = "Esperando"
    progress_data_gb: Optional[float] = None
    progress_target_gb: Optional[float] = None
    progress_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def update_job_progress(
    job_id: Optional[str],
    percent: float,
    label: str,
    *,
    data_gb: Optional[float] = None,
    target_gb: Optional[float] = None,
    step: Optional[str] = None,
) -> None:
    if not job_id:
        return
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.progress_percent = max(0.0, min(100.0, float(percent)))
        job.progress_label = label
        job.progress_data_gb = data_gb
        job.progress_target_gb = target_gb
        job.progress_step = step
        job.updated_at = time.time()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_model_catalog() -> Dict[str, Dict[str, str]]:
    ordered = {key: pipeline.MODEL_CATALOG[key] for key in MODEL_SERIES_ORDER if key in pipeline.MODEL_CATALOG}
    for key, value in pipeline.MODEL_CATALOG.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def namespace_from_payload(payload: dict, dataset: str) -> SimpleNamespace:
    max_data_gb = payload.get("max_data_gb")
    max_data_percent = payload.get("max_data_percent")
    if max_data_percent not in (None, "") and max_data_gb in (None, ""):
        percent = float(max_data_percent)
        if percent <= 0 or percent > 100:
            raise ValueError("max_data_percent debe estar entre 0 y 100.")
        config = pipeline.resolve_dataset(dataset)
        total_bytes = dataset_size_summary(config)["data_size_bytes"]
        max_data_bytes = int(total_bytes * (percent / 100.0))
        max_data_gb = max_data_bytes / (1024 ** 3)
    else:
        max_data_bytes = int(float(max_data_gb) * 1024 * 1024 * 1024) if max_data_gb not in (None, "") else None
    budget_active = max_data_bytes is not None
    return SimpleNamespace(
        dataset=dataset,
        model_kind=payload.get("model_kind") or None,
        limit_files=None if budget_active else payload.get("limit_files") or None,
        max_files_per_class=None if budget_active else payload.get("max_files_per_class") or None,
        max_data_gb=float(max_data_gb) if max_data_gb not in (None, "") else None,
        max_data_percent=float(max_data_percent) if max_data_percent not in (None, "") else None,
        max_data_bytes=max_data_bytes,
        ieee_target=payload.get("ieee_target") or "auto",
        window_strategy=payload.get("window_strategy") or "linspace",
        window_size=int(payload.get("window_size") or 4096),
        windows_per_file=int(payload.get("windows_per_file") or 3),
        test_size=float(payload.get("test_size") or 0.25),
        random_state=int(payload.get("random_state") or 42),
    )


def validation_payload(dataset: str, model_kind: Optional[str] = None) -> Dict[str, Any]:
    path = pipeline.REPORTS_DIR / f"{dataset}_{model_kind}_validation.json" if model_kind else pipeline.REPORTS_DIR / f"{dataset}_validation.json"
    report = read_json(path)
    if report and not report.get("model_size_bytes"):
        model_path_value = report.get("model_path")
        model_path = ROOT / model_path_value if model_path_value else None
        if model_path and model_path.exists():
            report["model_size_bytes"] = model_path.stat().st_size
    return {"path": rel(path), "report": report}


def inventory_payload(dataset: str) -> Dict[str, Any]:
    path = pipeline.REPORTS_DIR / f"{dataset}_inventory.json"
    report = read_json(path)
    return {"path": rel(path), "report": report}


def benchmark_payload(dataset: str) -> Dict[str, Any]:
    path = pipeline.REPORTS_DIR / f"{dataset}_benchmark.json"
    report = read_json(path)
    if report and "benchmark_path" not in report:
        report["benchmark_path"] = rel(path)
    return {"path": rel(path), "report": report}


def model_registry(dataset: Optional[str] = None) -> Dict[str, Any]:
    items = []
    dataset_names = [dataset] if dataset else list(pipeline.DATASETS)
    for matched_dataset in dataset_names:
        if not matched_dataset or matched_dataset not in pipeline.DATASETS:
            continue
        for matched_model in ordered_model_catalog():
            model_path = pipeline.MODELS_DIR / f"{matched_dataset}_{matched_model}_model.joblib"
            validation_path = pipeline.REPORTS_DIR / f"{matched_dataset}_{matched_model}_validation.json"
            model_exists = model_path.exists()
            validation = read_json(validation_path) or {}
            items.append(
                {
                    "dataset": matched_dataset,
                    "model_kind": matched_model,
                    "model_path": rel(model_path),
                "model_exists": model_exists,
                "model_size_bytes": (model_path.stat().st_size if model_exists else None) or validation.get("model_size_bytes"),
                    "updated_at": model_path.stat().st_mtime if model_exists else None,
                    "validation_path": rel(validation_path),
                    "records_used": validation.get("records_used"),
                    "data_size_bytes": validation.get("data_size_bytes"),
                    "referenced_data_size_bytes": validation.get("referenced_data_size_bytes") or validation.get("data_size_bytes"),
                    "estimated_iq_bytes_read": validation.get("estimated_iq_bytes_read"),
                    "estimated_iq_gb_read": validation.get("estimated_iq_gb_read"),
                    "requested_max_data_percent": validation.get("requested_max_data_percent"),
                    "requested_max_data_gb": validation.get("requested_max_data_gb"),
                    "requested_max_data_bytes": validation.get("requested_max_data_bytes"),
                    "windows": validation.get("windows"),
                    "classes": len(validation.get("classes") or []),
                    "holdout_accuracy": validation.get("holdout_accuracy"),
                    "balanced_accuracy": validation.get("balanced_accuracy"),
                    "macro_f1": validation.get("macro_f1"),
                    "train_records": validation.get("train_records"),
                    "holdout_records_count": validation.get("holdout_records_count"),
                    "split_protocol": validation.get("split_protocol"),
                }
            )
    return {"models": items}


def summarize_record(record: dict) -> dict:
    return {
        "meta_path": rel(record["meta_path"]),
        "data_path": rel(record["data_path"]),
        "label": record["label"],
        "dtype": record["dtype"],
        "sample_count": record["sample_count"],
        "data_size_bytes": record.get("data_size_bytes"),
        "group": record["group"],
    }


def dataset_size_summary(config: pipeline.DatasetConfig) -> Dict[str, Any]:
    data_files = []
    first_by_label: Dict[str, int] = {}
    meta_count = 0
    for meta_path in sorted(config.root.rglob(config.metadata_glob)):
        data_path = pipeline.data_path_for_meta(config, meta_path)
        if data_path.exists():
            size = data_path.stat().st_size
            if config.name == "kri_wifi":
                match = pipeline.re.search(r"WiFi_air_X310_([^_]+)_", meta_path.name)
                label = match.group(1) if match else "unknown"
            elif config.name == "uav_lightbridge":
                match = pipeline.re.search(r"(uav\d+)_", meta_path.name)
                label = match.group(1) if match else "unknown"
            elif config.name == "ieee_cbrs":
                label = meta_path.parent.name
            elif config.name == "wifi_dat_day":
                label = next(
                    (p.name for p in meta_path.parents if pipeline.re.fullmatch(r"Device_\d+", p.name, flags=pipeline.re.IGNORECASE)),
                    "unknown_dat",
                )
            else:
                label = "unknown"
            if label not in first_by_label:
                first_by_label[label] = size
            meta_count += 1
            data_files.append(data_path)
    total_bytes = sum(p.stat().st_size for p in data_files)
    min_balanced_bytes = sum(first_by_label.values())
    return {
        "records": meta_count,
        "data_files": len(data_files),
        "data_size_bytes": total_bytes,
        "data_size_gb": total_bytes / (1024 ** 3),
        "classes": len(first_by_label),
        "min_balanced_bytes": min_balanced_bytes,
        "min_balanced_gb": min_balanced_bytes / (1024 ** 3),
    }


def run_discover(payload: dict) -> Dict[str, Any]:
    dataset = payload["dataset"]
    config = pipeline.resolve_dataset(dataset)
    args = namespace_from_payload(payload, dataset)
    out = pipeline.write_inventory(config, args)
    inv = read_json(out) or {}
    return {"inventory_path": rel(out), "inventory": inv}


def run_train_like(payload: dict, operation: str) -> Dict[str, Any]:
    dataset = payload["dataset"]
    config = pipeline.resolve_dataset(dataset)
    args = namespace_from_payload(payload, dataset)
    job_id = payload.get("_job_id")
    target_gb = (args.max_data_bytes / (1024 ** 3)) if args.max_data_bytes else None
    update_job_progress(job_id, 5, "Preparando seleccion de capturas", target_gb=target_gb, step=operation)
    start = time.perf_counter()
    update_job_progress(job_id, 20, "Extrayendo ventanas IQ y entrenando modelo", target_gb=target_gb, step=operation)
    model_path = pipeline.train_dataset(config, args)
    elapsed = time.perf_counter() - start
    report = validation_payload(dataset, args.model_kind or config.model_kind)
    validation = report["report"] or {}
    used_gb = float((validation.get("referenced_data_size_bytes") or 0) / (1024 ** 3))
    update_job_progress(job_id, 100, "Entrenamiento completado", data_gb=used_gb, target_gb=target_gb, step=operation)
    return {
        "model_path": rel(model_path),
        "model_kind": args.model_kind or config.model_kind,
        "validation": report,
        "operation": operation,
        "time_seconds": elapsed,
    }


def run_compare(payload: dict) -> Dict[str, Any]:
    dataset = payload["dataset"]
    config = pipeline.resolve_dataset(dataset)
    size_summary = dataset_size_summary(config)
    dataset_total_bytes = size_summary["data_size_bytes"]
    requested = payload.get("model_kinds") or payload.get("models") or []
    model_kinds = requested or list(ordered_model_catalog())
    prediction_limit = int(payload.get("prediction_limit") or 5)
    job_id = payload.get("_job_id")
    first_args = namespace_from_payload(payload, dataset)
    target_gb = (first_args.max_data_bytes / (1024 ** 3)) if first_args.max_data_bytes else None
    results = []
    total_steps = max(1, len(model_kinds) * 3)
    done_steps = 0
    for model_index, model_kind in enumerate(model_kinds, start=1):
        train_args = namespace_from_payload({**payload, "model_kind": model_kind}, dataset)
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Train inicial: {model_kind}",
            target_gb=target_gb,
            step="train",
        )
        train_start = time.perf_counter()
        train_model_path = pipeline.train_dataset(config, train_args)
        train_time_seconds = time.perf_counter() - train_start
        train_report = validation_payload(dataset, model_kind)["report"] or {}
        used_gb = float((train_report.get("referenced_data_size_bytes") or 0) / (1024 ** 3))
        done_steps += 1
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Train completado: {model_kind}",
            data_gb=used_gb,
            target_gb=target_gb,
            step="train",
        )

        retrain_payload = {**payload, "model_kind": model_kind, "random_state": int(payload.get("random_state") or 42) + 1000}
        retrain_args = namespace_from_payload(retrain_payload, dataset)
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Reentrenando: {model_kind}",
            data_gb=used_gb,
            target_gb=target_gb,
            step="retrain",
        )
        retrain_start = time.perf_counter()
        model_path = pipeline.train_dataset(config, retrain_args)
        retrain_time_seconds = time.perf_counter() - retrain_start
        retrain_report = validation_payload(dataset, model_kind)["report"] or {}
        used_gb = float((retrain_report.get("referenced_data_size_bytes") or 0) / (1024 ** 3))
        done_steps += 1
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Reentrenamiento completado: {model_kind}",
            data_gb=used_gb,
            target_gb=target_gb,
            step="retrain",
        )

        bundle = joblib.load(model_path)
        holdout_records = retrain_report.get("holdout_records") or []
        if holdout_records:
            prediction_records = [
                {
                    "data_path": ROOT / record["data_path"],
                    "label": record["label"],
                    "dtype": record["dtype"],
                    "sample_count": record.get("sample_count"),
                    "start_sample": record.get("start_sample"),
                    "end_sample": record.get("end_sample"),
                }
                for record in holdout_records[:prediction_limit]
            ]
        else:
            prediction_records = list(
                pipeline.iter_records(
                    config,
                    limit_files=None,
                    max_files_per_class=1,
                    ieee_target=payload.get("ieee_target") or "auto",
                )
            )[:prediction_limit]
        predictions = []
        prediction_hits = 0
        prediction_total_time_seconds = 0.0
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Prediccion de control: {model_kind}",
            data_gb=used_gb,
            target_gb=target_gb,
            step="predict",
        )
        for record in prediction_records:
            prediction_start = time.perf_counter()
            pred = pipeline.predict_record(
                bundle,
                record["data_path"],
                record["dtype"],
                record.get("sample_count"),
                top_k=3,
                start_sample=int(record.get("start_sample") or 0),
            )
            prediction_elapsed = time.perf_counter() - prediction_start
            prediction_total_time_seconds += prediction_elapsed
            hit = pred["prediction"] == record["label"]
            prediction_hits += int(hit)
            predictions.append(
                {
                    "file": rel(record["data_path"]),
                    "expected": record["label"],
                    "predicted": pred["prediction"],
                    "confidence": pred["confidence"],
                    "hit": hit,
                    "prediction_time_seconds": prediction_elapsed,
                    "top_k": pred.get("top_k", []),
                }
            )
        prediction_count = len(predictions)
        prediction_failures = prediction_count - prediction_hits
        prediction_accuracy = prediction_hits / prediction_count if prediction_count else None
        done_steps += 1
        update_job_progress(
            job_id,
            (done_steps / total_steps) * 100,
            f"[{model_index}/{len(model_kinds)}] Modelo evaluado: {model_kind}",
            data_gb=used_gb,
            target_gb=target_gb,
            step="rank",
        )
        mean_prediction_time_seconds = prediction_total_time_seconds / prediction_count if prediction_count else None
        train_macro = float(train_report.get("macro_f1") or 0)
        retrain_macro = float(retrain_report.get("macro_f1") or 0)
        stability_gap = abs(retrain_macro - train_macro)
        score = (
            0.35 * float(retrain_report.get("macro_f1") or 0)
            + 0.25 * float(retrain_report.get("balanced_accuracy") or 0)
            + 0.20 * float(retrain_report.get("holdout_accuracy") or 0)
            + 0.15 * float(prediction_accuracy or 0)
            + 0.05 * max(0.0, 1.0 - stability_gap)
        )
        results.append(
            {
                "model_kind": model_kind,
                "family": pipeline.MODEL_CATALOG[model_kind]["family"],
                "paper_link": pipeline.MODEL_CATALOG[model_kind]["paper_link"],
                "model_path": rel(model_path),
                "train": {
                    "model_path": rel(train_model_path),
                    "holdout_accuracy": train_report.get("holdout_accuracy"),
                    "balanced_accuracy": train_report.get("balanced_accuracy"),
                    "macro_f1": train_report.get("macro_f1"),
                    "time_seconds": train_time_seconds,
                },
                "retrain": {
                    "model_path": rel(model_path),
                    "holdout_accuracy": retrain_report.get("holdout_accuracy"),
                    "balanced_accuracy": retrain_report.get("balanced_accuracy"),
                    "macro_f1": retrain_report.get("macro_f1"),
                    "time_seconds": retrain_time_seconds,
                },
                "stability_gap_macro_f1": stability_gap,
                "prediction_accuracy": prediction_accuracy,
                "prediction_hits": prediction_hits,
                "prediction_failures": prediction_failures,
                "prediction_count": prediction_count,
                "prediction_total_time_seconds": prediction_total_time_seconds,
                "mean_prediction_time_seconds": mean_prediction_time_seconds,
                "predictions": predictions,
                "score": score,
                "total_benchmark_time_seconds": train_time_seconds + retrain_time_seconds + prediction_total_time_seconds,
                "records_used": retrain_report.get("records_used"),
                "data_size_bytes": retrain_report.get("data_size_bytes"),
                "referenced_data_size_bytes": retrain_report.get("referenced_data_size_bytes") or retrain_report.get("data_size_bytes"),
                "estimated_iq_bytes_read": retrain_report.get("estimated_iq_bytes_read"),
                "estimated_iq_gb_read": retrain_report.get("estimated_iq_gb_read"),
                "requested_max_data_percent": retrain_report.get("requested_max_data_percent"),
                "requested_max_data_gb": retrain_report.get("requested_max_data_gb"),
                "requested_max_data_bytes": retrain_report.get("requested_max_data_bytes"),
                "referenced_percent_of_total": (
                    float((retrain_report.get("referenced_data_size_bytes") or retrain_report.get("data_size_bytes") or 0) / dataset_total_bytes * 100)
                    if dataset_total_bytes
                    else None
                ),
                "windows": retrain_report.get("windows"),
                "features": retrain_report.get("features"),
                "classes": len(retrain_report.get("classes") or []),
                "holdout_records_count": retrain_report.get("holdout_records_count"),
            }
        )
    results.sort(key=lambda row: row["score"], reverse=True)
    for idx, row in enumerate(results, start=1):
        row["rank"] = idx
    summary = pipeline.write_csv_summary()
    benchmark = {
        "dataset": dataset,
        "data_budget": {
            "dataset_total_bytes": size_summary["data_size_bytes"],
            "dataset_total_gb": size_summary["data_size_gb"],
            "requested_max_data_percent": getattr(first_args, "max_data_percent", None),
            "requested_max_data_gb": getattr(first_args, "max_data_gb", None),
            "requested_max_data_bytes": getattr(first_args, "max_data_bytes", None),
        },
        "protocol": {
            "steps": [
                "train inicial por modelo",
                "retrain con nueva semilla para medir estabilidad",
                "prediccion sobre muestras reales etiquetadas",
                "conteo de aciertos/fallos y medicion de tiempo de inferencia",
                "ranking ponderado por macro-F1, balanced accuracy, accuracy, prediccion y estabilidad",
            ],
            "ranking_score": "0.35*macro_f1_retrain + 0.25*balanced_accuracy_retrain + 0.20*accuracy_retrain + 0.15*prediction_accuracy + 0.05*(1-stability_gap_macro_f1)",
            "prediction_limit": prediction_limit,
            "data_budget": "El presupuesto por GB/% se aplica sobre archivos IQ completos seleccionados por clase y tiene prioridad sobre Limite total y Max archivos por clase. GB referenciados es el peso completo de esos archivos; GB leidos est. es lo que realmente se lee al extraer ventanas IQ.",
            "split": "Split estratificado por clase y separado por archivo/captura para evitar fuga directa entre ventanas y mantener representacion de dispositivos en holdout",
            "prediction_control": "Las predicciones de control se toman de capturas holdout del split de reentrenamiento cuando estan disponibles.",
            "timing": "time.perf_counter medido en backend para entrenamiento, reentrenamiento e inferencia de control",
        },
        "results": results,
        "best_model": results[0] if results else None,
        "summary_path": rel(summary),
    }
    out = pipeline.REPORTS_DIR / f"{dataset}_benchmark.json"
    out.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    benchmark["benchmark_path"] = rel(out)
    update_job_progress(job_id, 100, "Comparacion cientifica completada", target_gb=target_gb, step="completed")
    return benchmark


def run_predict(payload: dict) -> Dict[str, Any]:
    model_path = ROOT / payload["model_path"]
    meta_path = ROOT / payload["meta_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata no encontrada: {meta_path}")
    bundle = joblib.load(model_path)
    dataset = bundle["dataset"]
    config = pipeline.DATASETS[dataset]
    if config.name == "wifi_dat_day":
        meta = {}
        data_path = ROOT / payload["data_path"] if payload.get("data_path") else meta_path
        dtype_name = payload.get("dtype") or config.default_dtype
        sample_count = data_path.stat().st_size // pipeline.complex_sample_nbytes(dtype_name)
        start_sample = int(payload.get("start_sample") or 0)
    else:
        meta = pipeline.load_json(meta_path)
        data_path = ROOT / payload["data_path"] if payload.get("data_path") else pipeline.data_path_for_meta(config, meta_path)
        dtype_name = payload.get("dtype") or pipeline.dtype_from_meta(config, meta)
        sample_count = pipeline.sample_count_from_meta(meta)
        start_sample = 0
    expected = None
    try:
        expected = pipeline.label_for_meta(config, meta_path, meta, ieee_target=bundle.get("ieee_target", "auto"))
    except Exception:
        expected = None
    pred = pipeline.predict_record(
        bundle,
        data_path,
        dtype_name,
        sample_count,
        top_k=int(payload.get("top_k") or 5),
        start_sample=start_sample,
    )
    return {
        "prediction": pred,
        "expected_from_metadata": expected,
        "meta_path": rel(meta_path),
        "data_path": rel(data_path),
        "model_path": rel(model_path),
        "dataset": dataset,
    }


def run_summary(_: dict) -> Dict[str, Any]:
    out = pipeline.write_csv_summary()
    rows = []
    if out.exists():
        lines = out.read_text(encoding="utf-8").splitlines()
        if lines:
            headers = lines[0].split(",")
            for line in lines[1:]:
                values = line.split(",")
                rows.append(dict(zip(headers, values)))
    return {"summary_path": rel(out), "rows": rows}


OPERATIONS: Dict[str, Callable[[dict], Dict[str, Any]]] = {
    "discover": run_discover,
    "train": lambda payload: run_train_like(payload, "train"),
    "retrain": lambda payload: run_train_like(payload, "retrain"),
    "compare": run_compare,
    "predict": run_predict,
    "summary": run_summary,
}


def start_job(operation: str, payload: dict) -> Job:
    if operation not in OPERATIONS:
        raise ValueError(f"Operacion no soportada: {operation}")
    job = Job(
        id=str(uuid.uuid4()),
        operation=operation,
        status="queued",
        created_at=time.time(),
        updated_at=time.time(),
        params=payload,
    )
    with JOBS_LOCK:
        JOBS[job.id] = job

    def worker() -> None:
        with JOBS_LOCK:
            job.status = "running"
            job.progress_percent = 1.0
            job.progress_label = "Iniciando operacion"
            job.updated_at = time.time()
        try:
            job_payload = {**payload, "_job_id": job.id}
            result = OPERATIONS[operation](job_payload)
            with JOBS_LOCK:
                job.status = "completed"
                job.progress_percent = 100.0
                job.progress_label = "Completado"
                job.result = result
                job.updated_at = time.time()
        except Exception as exc:
            with JOBS_LOCK:
                job.status = "failed"
                job.error = str(exc)
                job.traceback = traceback.format_exc()
                job.updated_at = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return job


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print("[rf-web]", format % args)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/datasets":
                self.send_json(self.datasets())
            elif path == "/api/model-catalog":
                self.send_json({"models": ordered_model_catalog()})
            elif path == "/api/models":
                self.send_json(model_registry(query.get("dataset", [None])[0]))
            elif path == "/api/reports":
                dataset = query.get("dataset", [None])[0]
                model_kind = query.get("model_kind", [None])[0]
                if not dataset:
                    raise ValueError("Falta dataset")
                self.send_json(
                    {
                        "inventory": inventory_payload(dataset),
                        "validation": validation_payload(dataset, model_kind),
                        "benchmark": benchmark_payload(dataset),
                        "models": model_registry(dataset),
                        "model_path": rel(pipeline.MODELS_DIR / f"{dataset}_{model_kind}_model.joblib") if model_kind else rel(pipeline.MODELS_DIR / f"{dataset}_device_model.joblib"),
                    }
                )
            elif path == "/api/benchmark":
                dataset = query.get("dataset", [None])[0]
                if not dataset:
                    raise ValueError("Falta dataset")
                self.send_json(benchmark_payload(dataset))
            elif path == "/api/samples":
                dataset = query.get("dataset", [None])[0]
                if not dataset:
                    raise ValueError("Falta dataset")
                limit = int(query.get("limit", ["50"])[0])
                config = pipeline.resolve_dataset(dataset)
                args = SimpleNamespace(
                    limit_files=None,
                    max_files_per_class=int(query.get("max_files_per_class", ["5"])[0]),
                    max_data_bytes=None,
                    ieee_target=query.get("ieee_target", ["auto"])[0],
                )
                records = list(
                    pipeline.iter_records(
                        config,
                        limit_files=args.limit_files,
                        max_files_per_class=args.max_files_per_class,
                        max_data_bytes=args.max_data_bytes,
                        ieee_target=args.ieee_target,
                    )
                )[:limit]
                self.send_json({"samples": [summarize_record(r) for r in records]})
            elif path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    if not job:
                        self.send_json({"error": "Job no encontrado"}, HTTPStatus.NOT_FOUND)
                        return
                    self.send_json(asdict(job))
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    self.send_json({"jobs": [asdict(job) for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]})
            else:
                super().do_GET()
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                operation = parsed.path.split("/")[-1]
                payload = self.read_body()
                job = start_job(operation, payload)
                self.send_json(asdict(job), HTTPStatus.ACCEPTED)
            else:
                self.send_json({"error": "Ruta no soportada"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def datasets(self) -> Dict[str, Any]:
        datasets = []
        for name, config in pipeline.DATASETS.items():
            model_files = sorted(pipeline.MODELS_DIR.glob(f"{name}_*_model.joblib"))
            model_path = pipeline.MODELS_DIR / f"{name}_device_model.joblib"
            validation = read_json(pipeline.REPORTS_DIR / f"{name}_validation.json")
            inventory = read_json(pipeline.REPORTS_DIR / f"{name}_inventory.json")
            size_summary = dataset_size_summary(config)
            datasets.append(
                {
                    "name": name,
                    "root": rel(config.root),
                    "model_kind": config.model_kind,
                    "default_dtype": config.default_dtype,
                    "size": size_summary,
                    "model_exists": model_path.exists(),
                    "model_path": rel(model_path),
                    "available_models": [
                        {
                            "model_kind": p.name.replace(f"{name}_", "").replace("_model.joblib", ""),
                            "path": rel(p),
                        }
                        for p in model_files
                        if not p.name.endswith("_device_model.joblib")
                    ],
                    "inventory_records": (inventory or {}).get("records"),
                    "classes": (validation or inventory or {}).get("classes"),
                    "holdout_accuracy": (validation or {}).get("holdout_accuracy"),
                    "validation_path": rel(pipeline.REPORTS_DIR / f"{name}_validation.json"),
                    "inventory_path": rel(pipeline.REPORTS_DIR / f"{name}_inventory.json"),
                }
            )
        return {"datasets": datasets}


def main() -> None:
    parser = pipeline.argparse.ArgumentParser(description="RF fingerprinting web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RF web app: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
