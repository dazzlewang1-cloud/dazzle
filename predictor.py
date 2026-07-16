from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "ExtraTrees"


FEATURE_LABELS = {
    "mFEP": "mFEP",
    "NIHSS": "NIHSS",
    "mTTS": "mTTS",
    "HMCA sign": "HMCA sign",
    "mTMAX": "mTMAX",
    "CS": "CS",
    "NLR": "NLR",
    "Onset-to-reperfusion time": "Onset-to-reperfusion time (min)",
    "mMTT": "mMTT",
    "WBC count": "WBC count",
    "mCBF": "mCBF",
    "mTTD": "mTTD",
    "Hypertension": "Hypertension",
    "Diabetes": "Diabetes",
    "MCE": "MCE",
}


FEATURE_HELP = {
    "HMCA sign": "Binary variable: 0 = No, 1 = Yes.",
    "Hypertension": "Binary variable: 0 = No, 1 = Yes.",
    "Diabetes": "Binary variable: 0 = No, 1 = Yes.",
    "MCE": "Binary input for the joint prognostic model: 0 = No, 1 = Yes.",
    "Onset-to-reperfusion time": "Enter minutes from onset to reperfusion.",
}


FEATURE_ORDER = [
    "mFEP",
    "NIHSS",
    "mTTS",
    "HMCA sign",
    "mTMAX",
    "CS",
    "NLR",
    "Onset-to-reperfusion time",
    "mMTT",
    "WBC count",
    "mCBF",
    "mTTD",
    "Hypertension",
    "Diabetes",
]


FEATURE_SECTIONS = [
    ("Shared Clinical and Imaging Features", ["mFEP", "NIHSS", "mTTS", "HMCA sign"]),
    ("Additional Features for the MCE Model", ["mTMAX", "CS", "NLR", "Onset-to-reperfusion time"]),
    (
        "Additional Features for the Functional Outcome Model",
        ["mMTT", "WBC count", "mCBF", "mTTD", "Hypertension", "Diabetes"],
    ),
]


MODEL_SPECS = {
    "mce": {
        "label": "Malignant Cerebral Edema Model",
        "directory": APP_DIR / "assets" / "mce",
    },
    "functional": {
        "label": "Three-Month Functional Outcome Model",
        "directory": APP_DIR / "assets" / "diagnosis",
    },
    "functional_mce": {
        "label": "Three-Month Functional Outcome Model with MCE",
        "directory": APP_DIR / "assets" / "diagnosis_mce",
    },
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    help_text: str
    widget: str
    default: int | float
    options: tuple[int, ...] | None
    step: int | float
    format_string: str


@dataclass(frozen=True)
class ModelBundle:
    key: str
    label: str
    directory: Path
    model: Any
    feature_names: tuple[str, ...]
    threshold: float
    dca_threshold: float | None
    train_frame: pd.DataFrame


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probability: float
    threshold: float
    high_risk: bool


@dataclass(frozen=True)
class PredictionSummary:
    mce: PredictionResult
    functional: PredictionResult
    functional_mce: PredictionResult
    mce_input_for_joint_model: int
    mce_input_source: str


def _load_threshold(base_dir: Path, model_name: str) -> float:
    performance = pd.read_csv(base_dir / "final_model_performance.csv")
    row = performance.loc[performance["Model"] == model_name]
    if row.empty:
        raise ValueError(f"{model_name} threshold was not found in {base_dir}/final_model_performance.csv")
    return float(row.iloc[0]["Threshold"])


def _load_dca_threshold(base_dir: Path, model_name: str) -> float | None:
    threshold_path = base_dir / "optimal_thresholds.csv"
    if not threshold_path.exists():
        return None

    thresholds = pd.read_csv(threshold_path)
    row = thresholds.loc[thresholds["Model"] == model_name]
    if row.empty:
        return None
    return float(row.iloc[0]["Optimal_Threshold"])


def _load_bundle(key: str, label: str, directory: Path) -> ModelBundle:
    model = joblib.load(directory / "best_model.pkl")
    feature_info = joblib.load(directory / "feature_info.pkl")
    train_frame = pd.read_csv(directory / "train_data_final.csv")
    return ModelBundle(
        key=key,
        label=label,
        directory=directory,
        model=model,
        feature_names=tuple(feature_info["feature_names"]),
        threshold=_load_threshold(directory, MODEL_NAME),
        dca_threshold=_load_dca_threshold(directory, MODEL_NAME),
        train_frame=train_frame,
    )


def load_bundles() -> dict[str, ModelBundle]:
    return {
        key: _load_bundle(key=key, label=spec["label"], directory=spec["directory"])
        for key, spec in MODEL_SPECS.items()
    }


def _infer_feature_spec(name: str, series: pd.Series) -> FeatureSpec:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"Feature {name} has no numeric training values available.")

    options: tuple[int, ...] | None = None
    widget = "float"
    step: int | float = 0.01
    format_string = "%.2f"

    as_int = numeric.round().equals(numeric)
    unique_sorted = tuple(int(v) for v in sorted(numeric.unique())) if as_int else ()

    if as_int and unique_sorted in {(0, 1), (0,), (1,)}:
        widget = "binary"
        options = (0, 1)
        default = int(round(float(numeric.mode().iloc[0])))
        step = 1
        format_string = "%d"
    elif as_int and len(unique_sorted) <= 6:
        widget = "select"
        options = unique_sorted
        default = int(round(float(numeric.mode().iloc[0])))
        step = 1
        format_string = "%d"
    elif as_int:
        widget = "int"
        default = int(round(float(numeric.median())))
        step = 1
        format_string = "%d"
    else:
        widget = "float"
        default = float(numeric.median())
        step = 0.01
        format_string = "%.2f"

    return FeatureSpec(
        name=name,
        label=FEATURE_LABELS.get(name, name),
        help_text=FEATURE_HELP.get(name, ""),
        widget=widget,
        default=default,
        options=options,
        step=step,
        format_string=format_string,
    )


def build_feature_specs(bundles: dict[str, ModelBundle]) -> dict[str, FeatureSpec]:
    feature_specs: dict[str, FeatureSpec] = {}

    for name in FEATURE_ORDER + ["MCE"]:
        for bundle in bundles.values():
            if name in bundle.train_frame.columns:
                feature_specs[name] = _infer_feature_spec(name, bundle.train_frame[name])
                break
        else:
            raise ValueError(f"Feature {name} was not found in any training frame.")

    return feature_specs


def prepare_model_input(values: dict[str, int | float], bundle: ModelBundle) -> pd.DataFrame:
    row = {feature: values[feature] for feature in bundle.feature_names}
    return pd.DataFrame([row], columns=bundle.feature_names)


def predict_probability(values: dict[str, int | float], bundle: ModelBundle) -> float:
    model_input = prepare_model_input(values, bundle)
    return float(bundle.model.predict_proba(model_input)[0, 1])


def classify_risk(probability: float, threshold: float) -> bool:
    return probability >= threshold


def run_prediction(
    values: dict[str, int | float],
    bundles: dict[str, ModelBundle],
    actual_mce: int | None = None,
) -> PredictionSummary:
    mce_bundle = bundles["mce"]
    functional_bundle = bundles["functional"]
    functional_mce_bundle = bundles["functional_mce"]

    mce_probability = predict_probability(values, mce_bundle)
    functional_probability = predict_probability(values, functional_bundle)

    if actual_mce is None:
        mce_for_joint = int(classify_risk(mce_probability, mce_bundle.threshold))
        mce_source = "Auto-filled from the MCE model classification"
    else:
        mce_for_joint = int(actual_mce)
        mce_source = "Manual MCE input"

    joint_values = dict(values)
    joint_values["MCE"] = mce_for_joint
    functional_mce_probability = predict_probability(joint_values, functional_mce_bundle)

    return PredictionSummary(
        mce=PredictionResult(
            label="MCE probability",
            probability=mce_probability,
            threshold=mce_bundle.threshold,
            high_risk=classify_risk(mce_probability, mce_bundle.threshold),
        ),
        functional=PredictionResult(
            label="3-month poor functional outcome probability",
            probability=functional_probability,
            threshold=functional_bundle.threshold,
            high_risk=classify_risk(functional_probability, functional_bundle.threshold),
        ),
        functional_mce=PredictionResult(
            label="3-month poor functional outcome probability (joint model)",
            probability=functional_mce_probability,
            threshold=functional_mce_bundle.threshold,
            high_risk=classify_risk(functional_mce_probability, functional_mce_bundle.threshold),
        ),
        mce_input_for_joint_model=mce_for_joint,
        mce_input_source=mce_source,
    )
