"""
Evaluation script for resume model.

STRICT CONSTRAINTS
- Does not modify existing files.
- Writes only to results/ directory (auto-created).
- Reads data from data/resume/ by default.
- Loads architecture from Architecture.py (no redefinition here).

Features
- Robust model loading across common frameworks (PyTorch, Keras, scikit-learn).
- Safe weight loading if available, with graceful fallbacks.
- Resume data ingestion from folder structure or labels.csv.
- Text extraction for .txt, .pdf, .docx with graceful library fallbacks.
- Comprehensive metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC (if applicable).
- High-quality plots (PNG, 300 DPI) for metrics and curves.
- Architecture visualization (framework-specific when possible, otherwise textual schematic).
- Clear logging, modular functions, and fault tolerance.

CLI
python evaluate_model.py --data-dir data/resume --results-dir results --weights <optional>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional third-party imports handled lazily

# Plotting
import matplotlib
matplotlib.use("Agg")  # Ensure no GUI backend is required
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Data utilities
import csv
import re

LOGGER = logging.getLogger("evaluate_model")

# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class EvalConfig:
    data_dir: Path
    results_dir: Path
    weights_path: Optional[Path] = None
    batch_size: int = 16
    max_files: Optional[int] = None
    history_file: Optional[Path] = None
    seed: int = 42


# -----------------------------
# Utility: Setup
# -----------------------------

def setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Architecture import/loading
# -----------------------------

class Framework:
    TORCH = "torch"
    KERAS = "keras"
    SKLEARN = "sklearn"
    UNKNOWN = "unknown"


def safe_import_architecture() -> Any:
    """Import Architecture.py robustly without redefining architecture here."""
    module_name = "Architecture"
    try:
        # Try standard import
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Try relative path import from current working dir
        arch_path = Path.cwd() / "Architecture.py"
        if arch_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(arch_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                return module
        raise


def detect_framework(model: Any) -> str:
    try:
        import torch
        import torch.nn as nn  # noqa: F401
        if hasattr(model, "__class__") and any(
            base.__module__.startswith("torch.nn") for base in inspect.getmro(model.__class__)
        ):
            return Framework.TORCH
    except Exception:
        pass

    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return Framework.KERAS
    except Exception:
        pass

    # Heuristic for scikit-learn estimators/pipelines
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return Framework.SKLEARN
    except Exception:
        pass

    return Framework.UNKNOWN


def _can_instantiate_without_args(cls: Any) -> bool:
    try:
        sig = inspect.signature(cls)
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                return False
        return True
    except Exception:
        return False


def instantiate_model(arch_module: Any, preferred: Optional[str] = None) -> Any:
    """Instantiate model from Architecture.py.
    Strategy:
    - If --arch-object provided, use that name (variable/func/class)
    - Prefer callable functions: load_model(), get_model(), build_model(), make_model(), create_model()
    - Try common variables holding instances: model, MODEL, net, clf, pipeline, estimator, classifier
    - Try class names: Model, Net, Architecture; or first framework-looking class instantiable without args
    """
    # If a preferred symbol name was provided, use it first
    if preferred:
        obj = getattr(arch_module, preferred, None)
        if obj is None:
            LOGGER.warning("Preferred arch object '%s' not found in Architecture.py", preferred)
        else:
            # If it's callable, call it with no args; else if it's an instance, return directly; if class, try to instantiate
            if callable(obj) and not inspect.isclass(obj):
                LOGGER.info("Instantiating model via preferred function %s()", preferred)
                return obj()
            if inspect.isclass(obj):
                if _can_instantiate_without_args(obj):
                    LOGGER.info("Instantiating model via preferred class %s", preferred)
                    return obj()
            else:
                LOGGER.info("Using preferred model instance '%s'", preferred)
                return obj

    # Function-based builders
    for fn_name in ("load_model", "get_model", "build_model", "make_model", "create_model"):
        fn = getattr(arch_module, fn_name, None)
        if callable(fn):
            LOGGER.info("Instantiating model via %s()", fn_name)
            return fn()

    # Pre-instantiated variables
    for var_name in ("model", "MODEL", "net", "clf", "pipeline", "estimator", "classifier"):
        obj = getattr(arch_module, var_name, None)
        if obj is None:
            continue
        if callable(obj) and not inspect.isclass(obj):
            try:
                LOGGER.info("Instantiating model via callable variable %s()", var_name)
                return obj()
            except Exception:
                continue
        if not inspect.isclass(obj):
            LOGGER.info("Using model instance from variable '%s'", var_name)
            return obj

    # Class-based common names
    for cls_name in ("Model", "Net", "Architecture"):
        cls = getattr(arch_module, cls_name, None)
        if inspect.isclass(cls) and _can_instantiate_without_args(cls):
            LOGGER.info("Instantiating model via class %s", cls_name)
            return cls()

    # Fallback: first instantiable class in module that looks like a model
    for name, obj in inspect.getmembers(arch_module, inspect.isclass):
        try:
            if obj.__module__ != arch_module.__name__:
                continue
            if not _can_instantiate_without_args(obj):
                continue
            # Quick framework hint: prefer common base names
            bases = " ".join([b.__name__ for b in obj.__mro__])
            if any(k in bases.lower() for k in ("module", "sequential", "model", "estimator", "pipeline")):
                LOGGER.info("Instantiating model via discovered class %s", name)
                return obj()
        except Exception:
            continue

    # Log available symbols to help troubleshooting
    symbols = [n for n in dir(arch_module) if not n.startswith("__")]
    LOGGER.error("Available symbols in Architecture.py: %s", symbols)
    raise RuntimeError("No suitable model constructor found in Architecture.py. Consider exposing a 'load_model()' or 'get_model()' or a 'model' instance, or pass --arch-object <name>.")


def find_weights_file(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        return explicit if explicit.exists() else None

    candidates: List[Path] = []
    search_dirs = [
        Path("checkpoints"), Path("models"), Path("artifacts"), Path("weights"), Path("."),
    ]
    exts = [".pt", ".pth", ".bin", ".ckpt", ".joblib", ".pkl", ".h5"]
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for ext in exts:
                candidates.extend(d.glob(f"**/*{ext}"))
    if not candidates:
        return None
    # Pick the latest modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_weights_if_available(model: Any, framework: str, weights_path: Optional[Path]) -> Any:
    if not weights_path:
        LOGGER.warning("No weights file found; evaluating with model's current state.")
        return model

    LOGGER.info("Loading weights from %s", weights_path)
    try:
        if framework == Framework.TORCH:
            import torch
            state = torch.load(str(weights_path), map_location="cpu")
            # Common patterns: plain state_dict or dict with 'state_dict'
            if isinstance(state, dict) and any(k in state for k in ("state_dict", "model_state", "model")):
                for key in ("state_dict", "model_state", "model"):
                    if key in state:
                        state = state[key]
                        break
            model.load_state_dict(state)
            return model
        elif framework == Framework.KERAS:
            # Keras loads weights via load_weights; full models via tf.keras.models.load_model
            try:
                import tensorflow as tf
                if weights_path.suffix.lower() in (".h5", ".keras"):
                    try:
                        model.load_weights(str(weights_path))
                    except Exception:
                        # Try loading full model if weights-only fails
                        loaded = tf.keras.models.load_model(str(weights_path))
                        model = loaded
                else:
                    model.load_weights(str(weights_path))
            except Exception:
                LOGGER.exception("Failed to load Keras weights.")
            return model
        elif framework == Framework.SKLEARN:
            # Load serialized estimator
            try:
                import joblib
            except Exception:
                from joblib import load as joblib_load  # type: ignore
                def joblib_load_wrapper(path):
                    return joblib_load(path)
                obj = joblib_load_wrapper(str(weights_path))
            else:
                obj = joblib.load(str(weights_path))
            # If serialized model returns a new model, prefer it
            model = obj
            return model
        else:
            LOGGER.warning("Unknown framework; skipping weight load.")
    except Exception:
        LOGGER.exception("Failed to load weights; continuing with defaults.")
    return model


# -----------------------------
# Preprocessing / Data loading
# -----------------------------

SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_file(path: Path) -> Optional[str]:
    """Extract text from a resume file with graceful degradation."""
    try:
        if path.suffix.lower() == ".txt":
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.suffix.lower() == ".pdf":
            # Try pdfplumber first, then PyPDF2
            try:
                import pdfplumber
                with pdfplumber.open(str(path)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return clean_text("\n".join(pages))
            except Exception:
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(path))
                    pages = [page.extract_text() or "" for page in reader.pages]
                    return clean_text("\n".join(pages))
                except Exception:
                    LOGGER.warning("PDF extraction failed for %s", path)
                    return None
        elif path.suffix.lower() == ".docx":
            try:
                import docx  # python-docx
                doc = docx.Document(str(path))
                text = "\n".join([p.text for p in doc.paragraphs])
                return clean_text(text)
            except Exception:
                LOGGER.warning("DOCX extraction failed for %s", path)
                return None
        else:
            return None
    except Exception:
        LOGGER.exception("Failed to read %s", path)
        return None


def discover_labeled_files(data_dir: Path, max_files: Optional[int] = None) -> Tuple[List[Path], List[str]]:
    """Support two labeling strategies:
    1) Subfolder per class: data/resume/<label>/*.ext
    2) labels.csv in data/resume with columns: filename,label
    Returns (paths, labels)
    """
    paths: List[Path] = []
    labels: List[str] = []

    labels_csv = data_dir / "labels.csv"
    if labels_csv.exists():
        try:
            with labels_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename")
                    label = row.get("label")
                    if not fname or not label:
                        continue
                    p = data_dir / fname
                    if p.exists() and p.suffix.lower() in SUPPORTED_EXTS:
                        paths.append(p)
                        labels.append(str(label))
                        if max_files and len(paths) >= max_files:
                            break
        except Exception:
            LOGGER.exception("Failed to parse labels.csv; falling back to folder structure.")

    if not paths:
        # Folder structure
        for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            label = sub.name
            for f in sorted(sub.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    paths.append(f)
                    labels.append(label)
                    if max_files and len(paths) >= max_files:
                        break
            if max_files and len(paths) >= max_files:
                break

    return paths, labels


def load_data(config: EvalConfig) -> Tuple[List[str], List[str], List[Path]]:
    if not config.data_dir.exists() or not any(config.data_dir.iterdir()):
        raise FileNotFoundError(f"Data directory is empty or missing: {config.data_dir}")

    file_paths, labels = discover_labeled_files(config.data_dir, max_files=config.max_files)
    if not file_paths:
        raise RuntimeError(
            f"No labeled resume files found under {config.data_dir}. Expected subfolders per class or labels.csv."
        )

    texts: List[str] = []
    final_paths: List[Path] = []
    final_labels: List[str] = []

    for p, y in zip(file_paths, labels):
        txt = extract_text_from_file(p)
        if txt is None:
            LOGGER.warning("Skipping unreadable file: %s", p)
            continue
        if not txt.strip():
            LOGGER.warning("Skipping empty text file: %s", p)
            continue
        texts.append(txt)
        final_paths.append(p)
        final_labels.append(y)

    if not texts:
        raise RuntimeError("All files failed to parse or are empty. Nothing to evaluate.")

    return texts, final_labels, final_paths


# -----------------------------
# Prediction helpers (framework-agnostic)
# -----------------------------

@dataclass
class PredictionOutput:
    y_true: List[str]
    y_pred: List[str]
    proba: Optional[List[List[float]]]  # shape: [n_samples, n_classes]
    classes_: List[str]


def get_classes_from_truth(y_true: Sequence[str]) -> List[str]:
    return sorted(set(y_true))


def predict_with_model(model: Any, framework: str, texts: List[str], y_true: List[str]) -> PredictionOutput:
    classes = get_classes_from_truth(y_true)

    # scikit-learn estimators and pipelines (preferred, handles its own vectorization)
    if framework == Framework.SKLEARN:
        y_pred = model.predict(texts)  # type: ignore[attr-defined]
        proba = None
        try:
            proba = model.predict_proba(texts).tolist()  # type: ignore[attr-defined]
        except Exception:
            proba = None
        try:
            # Prefer estimator-provided classes
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
        except Exception:
            pass
        return PredictionOutput(y_true=list(y_true), y_pred=list(map(str, y_pred)), proba=proba, classes_=classes)

    # Keras model: expect a tokenizer/preprocessor inside the Architecture module or model
    if framework == Framework.KERAS:
        # Try common preprocess hooks
        preproc = None
        for name in ("preprocess", "tokenize", "vectorize", "transform"):
            if hasattr(model, name) and callable(getattr(model, name)):
                preproc = getattr(model, name)
                break
        if preproc is None:
            LOGGER.warning("No Keras preprocessor found on model; attempting identity pass. Predictions may fail.")
            X = texts
        else:
            X = preproc(texts)
        # Predict
        import numpy as np
        preds = model.predict(X, verbose=0)
        if preds.ndim == 1 or preds.shape[-1] == 1:
            # Binary case
            proba = preds.reshape(-1, 1)
            y_pred = (preds.ravel() >= 0.5).astype(int)
            if len(classes) == 2:
                pass
            else:
                classes = ["0", "1"]
        else:
            proba = preds
            y_pred = preds.argmax(axis=-1)
        y_pred_str = [str(c) for c in y_pred]
        return PredictionOutput(y_true=list(y_true), y_pred=y_pred_str, proba=proba.tolist(), classes_=classes)

    # PyTorch model
    if framework == Framework.TORCH:
        import numpy as np
        import torch
        model.eval()
        # Try to find a tokenizer or preprocessor in Architecture module or model
        preproc = None
        for name in ("preprocess", "tokenize", "vectorize", "transform"):
            if hasattr(model, name) and callable(getattr(model, name)):
                preproc = getattr(model, name)
                break
        if preproc is None:
            LOGGER.warning("No PyTorch preprocessor found on model; falling back to naive token length embedding.")
            # Naive numeric feature: length of text. Model must accept a single float feature.
            # This is a last-resort fallback and likely not compatible with arbitrary architectures.
            # We'll still try to pass a tensor of shape [N, 1].
            X = np.array([[len(t)]] * len(texts), dtype=np.float32)
            X_tensor = torch.from_numpy(X)
        else:
            X = preproc(texts)
            if isinstance(X, np.ndarray):
                X_tensor = torch.from_numpy(X)
            else:
                # Assume list of tensors
                X_tensor = X  # type: ignore
        with torch.no_grad():
            logits = model(X_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            arr = logits.detach().cpu().numpy()
            if arr.ndim == 1 or arr.shape[-1] == 1:
                # Binary
                proba = 1 / (1 + np.exp(-arr.ravel()))
                y_pred = (proba >= 0.5).astype(int)
                proba = np.vstack([1 - proba, proba]).T
                if len(classes) != 2:
                    classes = ["0", "1"]
            else:
                # Multi-class softmax
                exp = np.exp(arr - arr.max(axis=-1, keepdims=True))
                proba = exp / exp.sum(axis=-1, keepdims=True)
                y_pred = proba.argmax(axis=-1)
        y_pred_str = [str(x) for x in y_pred.tolist()]
        return PredictionOutput(y_true=list(y_true), y_pred=y_pred_str, proba=proba.tolist(), classes_=classes)

    # Unknown framework: attempt scikit-learn like interface
    LOGGER.warning("Unknown framework; attempting generic predict/predict_proba.")
    y_pred = None
    proba = None
    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        try:
            y_pred = model.predict(texts)
        except Exception:
            LOGGER.exception("Generic predict() failed.")
    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        try:
            proba = model.predict_proba(texts).tolist()
        except Exception:
            proba = None
    if y_pred is None:
        raise RuntimeError("Model does not support prediction on provided data.")
    return PredictionOutput(y_true=list(y_true), y_pred=[str(x) for x in y_pred], proba=proba, classes_=classes)


# -----------------------------
# Metrics & Evaluation
# -----------------------------

@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion: List[List[int]]
    labels: List[str]
    roc_auc_macro: Optional[float] = None
    roc_auc_micro: Optional[float] = None


def compute_metrics(pred: PredictionOutput) -> Metrics:
    y_true = pred.y_true
    y_pred = pred.y_pred

    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro_, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_auc_macro = None
    roc_auc_micro = None
    # ROC-AUC requires probabilities and at least 2 classes
    if pred.proba is not None and len(labels) >= 2:
        y_true_bin = label_binarize(y_true, classes=labels)
        try:
            roc_auc_macro = roc_auc_score(y_true_bin, pred.proba, average="macro", multi_class="ovr")
            roc_auc_micro = roc_auc_score(y_true_bin, pred.proba, average="micro", multi_class="ovr")
        except Exception:
            LOGGER.warning("ROC-AUC computation not applicable; skipping.")

    return Metrics(
        accuracy=acc,
        precision_macro=prec_macro,
        recall_macro=rec_macro,
        f1_macro=f1_macro_,
        precision_weighted=prec_w,
        recall_weighted=rec_w,
        f1_weighted=f1_w,
        confusion=cm.tolist(),
        labels=labels,
        roc_auc_macro=roc_auc_macro,
        roc_auc_micro=roc_auc_micro,
    )


# -----------------------------
# Plotting utilities
# -----------------------------

sns.set_theme(style="whitegrid")


def plot_confusion_matrix_png(metrics: Metrics, out_path: Path) -> None:
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.heatmap(metrics.confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=metrics.labels, yticklabels=metrics.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_curve_png(pred: PredictionOutput, metrics: Metrics, out_path: Path) -> None:
    if pred.proba is None or len(metrics.labels) < 2:
        LOGGER.info("Skipping ROC curve plot (not applicable).")
        return
    try:
        y_true_bin = label_binarize(pred.y_true, classes=metrics.labels)
        import numpy as np
        proba = np.array(pred.proba)

        plt.figure(figsize=(8, 6), dpi=300)
        # Micro-average
        from sklearn.metrics import RocCurveDisplay
        if y_true_bin.shape[1] == 2:
            # Binary case: use positive class index 1
            fpr, tpr, _ = roc_curve(y_true_bin[:, 1], proba[:, 1])
            auc_val = metrics.roc_auc_macro if metrics.roc_auc_macro is not None else 0.0
            plt.plot(fpr, tpr, label=f"ROC (AUC macro={auc_val:.3f})")
        else:
            # Multiclass: plot micro-average
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), proba.ravel())
            plt.plot(fpr, tpr, label="ROC (micro-average)")
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
    except Exception:
        LOGGER.exception("Failed to plot ROC curve.")


def plot_history_or_placeholder(results_dir: Path, acc_path: Path, loss_path: Path, history_file: Optional[Path]) -> None:
    """Plot Accuracy vs Epochs and Loss vs Epochs.
    - If a history file is found (json/csv with 'accuracy'/'loss' lists), plot them.
    - Else, plot single-point placeholders with note in title.
    """
    history = None

    # Attempt to find history file
    def load_history_from(path: Path) -> Optional[Dict[str, List[float]]]:
        try:
            if not path.exists():
                return None
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                return {k: list(map(float, v)) for k, v in data.items() if isinstance(v, (list, tuple))}
            if path.suffix.lower() == ".csv":
                rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
                if not rows:
                    return None
                keys = [k for k in rows[0].keys() if k]
                out: Dict[str, List[float]] = {k: [] for k in keys}
                for row in rows:
                    for k in keys:
                        try:
                            out[k].append(float(row.get(k, 0) or 0))
                        except Exception:
                            pass
                return out
        except Exception:
            LOGGER.exception("Failed to parse history file: %s", path)
            return None
        return None

    # Explicit history
    if history_file:
        history = load_history_from(history_file)

    # Common default locations
    if history is None:
        for d in (Path("checkpoints"), Path("models"), Path("artifacts"), results_dir):
            for nm in ("history.json", "training_history.json", "history.csv", "training_history.csv"):
                hp = d / nm
                history = load_history_from(hp)
                if history:
                    break
            if history:
                break

    # Plot accuracy
    plt.figure(figsize=(8, 6), dpi=300)
    if history and any(k.lower().startswith("acc") for k in history.keys()):
        for k, v in history.items():
            if k.lower().startswith("acc"):
                plt.plot(range(1, len(v) + 1), v, label=k)
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    else:
        plt.plot([1], [0], marker="o")
        plt.title("Accuracy vs Epochs (no training history found)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 6), dpi=300)
    if history and any("loss" in k.lower() for k in history.keys()):
        for k, v in history.items():
            if "loss" in k.lower():
                plt.plot(range(1, len(v) + 1), v, label=k)
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
    else:
        plt.plot([1], [0], marker="o")
        plt.title("Loss vs Epochs (no training history found)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()


# -----------------------------
# Architecture diagramming
# -----------------------------

def has_convolution_layers_torch(model: Any) -> bool:
    try:
        import torch.nn as nn
        return any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
    except Exception:
        return False


def save_architecture_diagram(model: Any, framework: str, results_dir: Path) -> None:
    arch_png = results_dir / "architecture.png"
    conv_png = results_dir / "convolution.png"

    try:
        if framework == Framework.TORCH:
            try:
                # Try torchviz if available and if we can build a dummy input
                from torchviz import make_dot  # type: ignore
                import torch
                # Heuristic dummy input: prefer model.example_input_array if present
                example = getattr(model, "example_input_array", None) or getattr(model, "example_input", None)
                if example is None:
                    # Fallback to a simple [1, 10] tensor
                    example = torch.randn(1, 10)
                y = model(example)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                dot = make_dot(y, params=dict(model.named_parameters()))
                dot.format = "png"
                tmp_out = results_dir / "architecture"
                dot.render(filename=str(tmp_out))
                # torchviz writes architecture.png directly as tmp_out + ".png"
                rendered = tmp_out.with_suffix(".png")
                if rendered.exists():
                    rendered.replace(arch_png)
                else:
                    LOGGER.warning("torchviz did not produce expected PNG; falling back to textual diagram.")
            except Exception:
                LOGGER.warning("torchviz unavailable or failed; producing textual architecture diagram.")
                _save_textual_architecture(model, arch_png)

            # Convolution-specific diagram (if any conv layers)
            if has_convolution_layers_torch(model):
                try:
                    # Save a copy labeled for conv
                    if arch_png.exists():
                        import shutil
                        shutil.copy2(arch_png, conv_png)
                    else:
                        _save_textual_architecture(model, conv_png, title="Convolutional Layers Diagram")
                except Exception:
                    LOGGER.exception("Failed to export convolution.png")
            return

        if framework == Framework.KERAS:
            try:
                from tensorflow.keras.utils import plot_model  # type: ignore
                plot_model(model, to_file=str(arch_png), show_shapes=True, dpi=300)
            except Exception:
                LOGGER.warning("keras.utils.plot_model unavailable or failed; using textual diagram.")
                _save_textual_architecture(model, arch_png)

            # Convolution layers check
            try:
                import tensorflow as tf
                has_conv = any(
                    isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D))
                    for layer in model.layers
                )
                if has_conv:
                    # Duplicate architecture.png as convolution.png for simplicity
                    if arch_png.exists():
                        import shutil
                        shutil.copy2(arch_png, conv_png)
                    else:
                        _save_textual_architecture(model, conv_png, title="Convolutional Layers Diagram")
            except Exception:
                pass
            return

        # scikit-learn or unknown: textual diagram
        _save_textual_architecture(model, arch_png)
        # No strict conv concept; skip conv_png unless we can heuristically detect CNN keywords
    except Exception:
        LOGGER.exception("Failed to generate architecture diagram; writing textual fallback.")
        try:
            _save_textual_architecture(model, arch_png)
        except Exception:
            LOGGER.exception("Failed to write textual architecture as well.")


def _save_textual_architecture(model: Any, out_path: Path, title: str = "Model Architecture") -> None:
    plt.figure(figsize=(10, 8), dpi=300)
    plt.axis("off")
    text = _summarize_model_text(model)
    plt.title(title)
    plt.text(0.01, 0.99, text, va="top", ha="left", family="monospace", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _summarize_model_text(model: Any) -> str:
    buf = io.StringIO()
    buf.write(f"Type: {model.__class__.__name__}\n")
    buf.write("Attributes:\n")
    for k, v in list(vars(model).items())[:50]:
        try:
            s = str(v)
            if len(s) > 120:
                s = s[:117] + "..."
            buf.write(f"  - {k}: {s}\n")
        except Exception:
            pass
    try:
        source = inspect.getsource(model.__class__)
        if len(source) > 1500:
            source = source[:1500] + "\n... (truncated)"
        buf.write("\nClass Source (truncated):\n")
        buf.write(source)
    except Exception:
        buf.write("\n(Class source unavailable)\n")
    return buf.getvalue()


# -----------------------------
# Save metrics
# -----------------------------

def save_metrics(results_dir: Path, metrics: Metrics) -> None:
    json_path = results_dir / "metrics.json"
    txt_path = results_dir / "metrics.txt"

    data = {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "precision_weighted": metrics.precision_weighted,
        "recall_weighted": metrics.recall_weighted,
        "f1_weighted": metrics.f1_weighted,
        "labels": metrics.labels,
        "confusion_matrix": metrics.confusion,
        "roc_auc_macro": metrics.roc_auc_macro,
        "roc_auc_micro": metrics.roc_auc_micro,
    }

    try:
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        LOGGER.exception("Failed to write metrics.json")

    try:
        lines = [
            f"Accuracy: {metrics.accuracy:.4f}",
            f"Precision (macro): {metrics.precision_macro:.4f}",
            f"Recall (macro): {metrics.recall_macro:.4f}",
            f"F1 (macro): {metrics.f1_macro:.4f}",
            f"Precision (weighted): {metrics.precision_weighted:.4f}",
            f"Recall (weighted): {metrics.recall_weighted:.4f}",
            f"F1 (weighted): {metrics.f1_weighted:.4f}",
            f"Labels: {metrics.labels}",
            f"Confusion Matrix: {metrics.confusion}",
            f"ROC AUC (macro): {metrics.roc_auc_macro}",
            f"ROC AUC (micro): {metrics.roc_auc_micro}",
        ]
        txt_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        LOGGER.exception("Failed to write metrics.txt")


# -----------------------------
# Main evaluation flow
# -----------------------------

def evaluate(config: EvalConfig) -> None:
    ensure_results_dir(config.results_dir)

    # Load Architecture and instantiate model
    arch_module = safe_import_architecture()
    model = instantiate_model(arch_module, preferred=getattr(config, "arch_object", None))
    framework = detect_framework(model)
    LOGGER.info("Detected framework: %s", framework)

    # Load weights if found
    weights = find_weights_file(config.weights_path)
    if weights:
        LOGGER.info("Using weights file: %s", weights)
    else:
        LOGGER.warning("No weights file discovered.")
    model = load_weights_if_available(model, framework, weights)

    # Data
    texts, labels, paths = load_data(config)
    LOGGER.info("Loaded %d documents across %d classes", len(texts), len(set(labels)))

    # Predict
    pred = predict_with_model(model, framework, texts, labels)

    # Metrics
    metrics = compute_metrics(pred)

    # Print to console
    LOGGER.info("Accuracy: %.4f", metrics.accuracy)
    LOGGER.info("Precision (macro): %.4f | Recall (macro): %.4f | F1 (macro): %.4f",
                metrics.precision_macro, metrics.recall_macro, metrics.f1_macro)
    LOGGER.info("Precision (weighted): %.4f | Recall (weighted): %.4f | F1 (weighted): %.4f",
                metrics.precision_weighted, metrics.recall_weighted, metrics.f1_weighted)

    # Save metrics
    save_metrics(config.results_dir, metrics)

    # Plots
    plot_confusion_matrix_png(metrics, config.results_dir / "confusion_matrix.png")
    plot_roc_curve_png(pred, metrics, config.results_dir / "roc_curve.png")
    plot_history_or_placeholder(
        results_dir=config.results_dir,
        acc_path=config.results_dir / "accuracy.png",
        loss_path=config.results_dir / "loss.png",
        history_file=config.history_file,
    )

    # Architecture diagrams
    save_architecture_diagram(model, framework, config.results_dir)


# -----------------------------
# CLI entrypoint
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate resume model and export metrics + plots.")
    parser.add_argument("--data-dir", type=str, default="data/resume",
                        help="Directory containing resumes: subfolders per class or labels.csv.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Where to write results (will be created).")
    parser.add_argument("--weights", type=str, default=None,
                        help="Optional explicit path to weights or serialized model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (for frameworks that use it).")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files for quick runs.")
    parser.add_argument("--history-file", type=str, default=None,
                        help="Optional path to training history (json/csv) for accuracy/loss plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reserved for future use).")
    parser.add_argument("--arch-object", type=str, default=None,
                        help="Explicit name of function/class/variable inside Architecture.py to build/load the model (e.g., get_model, model, MyNet).")

    args = parser.parse_args(argv)
    cfg = EvalConfig(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        weights_path=Path(args.weights) if args.weights else None,
        batch_size=args.batch_size,
        max_files=args.max_files,
        history_file=Path(args.history_file) if args.history_file else None,
        seed=args.seed,
    )
    # Attach arch_object dynamically without changing dataclass signature for compatibility
    setattr(cfg, "arch_object", args.arch_object)
    return cfg


if __name__ == "__main__":
    setup_logging()
    try:
        cfg = parse_args()
        evaluate(cfg)
        LOGGER.info("Evaluation complete. Results saved under: %s", cfg.results_dir)
    except SystemExit:
        raise
    except Exception as e:
        LOGGER.error("Evaluation failed: %s", e)
        LOGGER.debug("\n" + traceback.format_exc())
        sys.exit(1)
