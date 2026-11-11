"""Package a fine-tuned detector as an MLflow pyfunc, smoke-test it, and register it in Unity Catalog.

Runs inside an AIR GPU workload (`recipes/<model>/register.yaml`). The serving environment is built from
pinned pip requirements (versions taken from the AIR environment). `env_pack="databricks_model_serving"`
is optional: for these classic pyfuncs it made endpoints crash-loop or time out (SPEC OQ-17).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import nemotron_dbx

SERVING_PARAMS = {"conf_threshold": -1.0, "apply_class_thresholds": True}


@dataclass
class RegistrationResult:
    model_name: str
    model_version: str
    model_uri: str
    smoke_boxes: int


def package_dir() -> Path:
    return Path(nemotron_dbx.__file__).resolve().parent


def build_detector_artifacts(spec_name: str, weights: str | Path, work_dir: Path, batch_size: int = 8):
    from ..detection.hf_detector import SPECS, load_exp, snapshot

    spec = SPECS[spec_name]
    code_dir = snapshot(spec, work_dir / "hf_code", with_weights=False)
    shutil.copy2(weights, work_dir / "weights.pth")
    exp = load_exp(spec, code_dir, device="cpu")
    config = {
        "spec": spec_name,
        "source": f"{spec.source.repo_id}@{spec.source.revision}",
        "labels": list(exp.labels),
        "thresholds_per_class": dict(exp.thresholds_per_class),
        "batch_size": batch_size,
    }
    (work_dir / "config.json").write_text(json.dumps(config, indent=2))
    return {
        "hf_code": str(code_dir),
        "weights": str(work_dir / "weights.pth"),
        "config": str(work_dir / "config.json"),
    }


def smoke_test_detector(artifacts: dict[str, str], sample_images: list[bytes]):
    import pandas as pd

    from ..detection.serving import DetectorModel

    model = DetectorModel()
    model.load_context(SimpleNamespace(artifacts=artifacts))
    df = pd.DataFrame([
        {"image": base64.b64encode(img).decode(), "id": f"smoke-{i}"} for i, img in enumerate(sample_images)
    ])
    out = model.predict(None, df, params=SERVING_PARAMS)
    if len(out) != len(sample_images) or not all("bounding_boxes" in o for o in out):
        raise RuntimeError(f"Unexpected smoke-test output: {str(out)[:500]}")
    n_boxes = sum(len(v) for o in out for v in o["bounding_boxes"].values())
    print(f"Smoke test OK: {len(out)} images, {n_boxes} boxes", flush=True)
    return df, out, n_boxes


def register_detector(
    spec_name: str,
    weights: str | Path,
    registered_model_name: str,
    sample_images: list[bytes],
    source_training_run_id: str | None = None,
    env_pack: bool = False,  # True crash-loops for classic pyfuncs on Model Serving (SPEC OQ-17)
    extra_tags: dict[str, str] | None = None,
) -> RegistrationResult:
    import mlflow
    from mlflow.models import infer_signature

    from ..detection.serving import DetectorModel

    mlflow.set_registry_uri("databricks-uc")
    os.environ.setdefault("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "true")
    with tempfile.TemporaryDirectory(prefix="nemotron-register-") as tmp:
        work = Path(tmp)
        artifacts = build_detector_artifacts(spec_name, weights, work)
        df, out, n_boxes = smoke_test_detector(artifacts, sample_images[:1])
        try:
            signature = infer_signature(df, out, params=SERVING_PARAMS)
        except Exception as exc:  # nested dict outputs: fall back to an input-only signature
            print(f"Output signature inference failed ({exc!r}); logging input/params signature only")
            signature = infer_signature(df, params=SERVING_PARAMS)

        run_id = os.environ.get("MLFLOW_RUN_ID")
        with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run() as run:
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=DetectorModel(),
                artifacts=artifacts,
                code_paths=[str(package_dir())],
                signature=signature,
                input_example=(df, SERVING_PARAMS),
                pip_requirements=_pip_requirements(),
            )
            kwargs = {"env_pack": "databricks_model_serving"} if env_pack else {}
            version = mlflow.register_model(info.model_uri, registered_model_name, **kwargs)
            tags = {
                "nemotron.spec": spec_name,
                "nemotron.registered_model": registered_model_name,
                "nemotron.registered_version": str(version.version),
                **({"nemotron.source_training_run_id": source_training_run_id} if source_training_run_id else {}),
                **(extra_tags or {}),
            }
            mlflow.set_tags(tags)
            client = mlflow.MlflowClient()
            if source_training_run_id:
                client.set_tag(source_training_run_id, "nemotron.registered_model", registered_model_name)
                client.set_tag(source_training_run_id, "nemotron.registered_version", str(version.version))
                client.set_tag(source_training_run_id, "nemotron.packaging_run_id", run.info.run_id)
    return RegistrationResult(registered_model_name, str(version.version), info.model_uri, n_boxes)


def _pip_requirements() -> list[str]:
    from importlib.metadata import version

    pkgs = ["mlflow", "torch", "torchvision", "numpy", "pillow", "pandas", "huggingface_hub"]
    reqs = []
    for p in pkgs:
        try:
            v = version(p)
        except Exception:  # noqa: BLE001
            reqs.append(p)
            continue
        reqs.append(f"{p}=={v.split('+')[0]}")
    return reqs
