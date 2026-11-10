"""Create or update a Model Serving endpoint for one exact UC model version, without waiting.

Adapted from ai-runtime-cookbook `serving/sft-qwen2.5-0.5b-lora-serving/deploy_endpoint.py`: idempotent
(re-running reports `unchanged`), refuses to clobber endpoints serving other models or with a different
update in flight.

    python -m nemotron_dbx.lifecycle.deploy --model-name cat.schema.model --model-version 3 \
        --endpoint-name nemotron-page-elements-v3-ft --workload-type GPU_SMALL --profile am-rag-fevm
"""

from __future__ import annotations

import argparse
import json


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _entity_matches(entity, desired) -> bool:
    return all([
        entity.entity_name == desired.entity_name,
        str(entity.entity_version) == str(desired.entity_version),
        _enum_value(entity.workload_type) == _enum_value(desired.workload_type),
        entity.workload_size == desired.workload_size,
        entity.scale_to_zero_enabled == desired.scale_to_zero_enabled,
    ])


def _served(config) -> list:
    return list(config.served_entities or []) if config else []


def submit(client, endpoint_name: str, desired) -> str:
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.serving import EndpointCoreConfigInput

    try:
        existing = client.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        client.serving_endpoints.create(
            name=endpoint_name, config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[desired])
        )
        return "created"

    current = _served(existing.config)
    pending = _served(existing.pending_config)
    state = existing.state
    in_progress = bool(state and state.config_update and _enum_value(state.config_update) == "IN_PROGRESS")
    if len(current) > 1:
        raise ValueError(f"Refusing to replace {endpoint_name}: it has multiple served entities.")
    observed = {e.entity_name for e in current + (pending if in_progress else []) if e.entity_name}
    if observed and observed != {desired.entity_name}:
        raise ValueError(f"Refusing to replace {endpoint_name}: it serves {sorted(observed)}.")
    if in_progress:
        if len(pending) == 1 and _entity_matches(pending[0], desired):
            return "already updating"
        raise RuntimeError(f"{endpoint_name} has a different update in progress; retry when it finishes.")
    if len(current) == 1 and _entity_matches(current[0], desired):
        return "unchanged"
    client.serving_endpoints.update_config(name=endpoint_name, served_entities=[desired])
    return "updated"


def main(argv: list[str] | None = None) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ServedEntityInput, ServingModelWorkloadType

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True, help="Three-level Unity Catalog model name")
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--workload-type", default="GPU_SMALL", help="e.g. CPU, GPU_SMALL, GPU_MEDIUM, GPU_LARGE")
    ap.add_argument("--workload-size", default="Small")
    ap.add_argument("--no-scale-to-zero", action="store_true")
    ap.add_argument("--profile")
    args = ap.parse_args(argv)

    desired = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=str(args.model_version),
        workload_type=ServingModelWorkloadType(args.workload_type),
        workload_size=args.workload_size,
        scale_to_zero_enabled=not args.no_scale_to_zero,
    )
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    action = submit(client, args.endpoint_name, desired)
    print("DEPLOYMENT_SUBMITTED=" + json.dumps({
        "action": action, "endpoint_name": args.endpoint_name,
        "model_name": args.model_name, "model_version": str(args.model_version),
    }, sort_keys=True))
    print(f"Check readiness: databricks serving-endpoints get {args.endpoint_name}"
          + (f" --profile {args.profile}" if args.profile else ""))


if __name__ == "__main__":
    main()
