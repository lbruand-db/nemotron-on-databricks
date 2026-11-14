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

# Workload types the API accepts but older databricks-sdk enums don't list. The SDK sends them fine (see
# `_RawWorkloadType`) but reads them back as None, so endpoint state is compared on the raw API JSON instead.
EXTRA_WORKLOAD_TYPES = ("GPU_LARGE_RTX",)


def _entity_key(e: dict) -> tuple:
    return (e.get("entity_name"), str(e.get("entity_version")), e.get("workload_type"), e.get("workload_size"),
            bool(e.get("scale_to_zero_enabled", False)))


def _entity_matches(entity: dict, desired: dict) -> bool:
    return _entity_key(entity) == _entity_key(desired)


class _RawWorkloadType(str):
    """A workload type the installed SDK's enum doesn't list yet (e.g. GPU_LARGE_RTX); serialises like the enum."""

    @property
    def value(self) -> str:
        return str(self)


def workload_type(name: str):
    from databricks.sdk.service.serving import ServingModelWorkloadType

    try:
        return ServingModelWorkloadType(name)
    except ValueError:
        if name in EXTRA_WORKLOAD_TYPES:
            return _RawWorkloadType(name)
        known = sorted({t.value for t in ServingModelWorkloadType} | set(EXTRA_WORKLOAD_TYPES))
        raise ValueError(f"Unknown workload type {name!r}; expected one of {known}") from None


def _served(config: dict | None) -> list[dict]:
    return list((config or {}).get("served_entities") or [])


def submit(client, endpoint_name: str, desired) -> str:
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.serving import EndpointCoreConfigInput

    try:
        existing = client.api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint_name}")
    except ResourceDoesNotExist:
        client.serving_endpoints.create(
            name=endpoint_name, config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[desired])
        )
        return "created"

    want = desired.as_dict()
    current = _served(existing.get("config"))
    pending = _served(existing.get("pending_config"))
    in_progress = (existing.get("state") or {}).get("config_update") == "IN_PROGRESS"
    if len(current) > 1:
        raise ValueError(f"Refusing to replace {endpoint_name}: it has multiple served entities.")
    observed = {e["entity_name"] for e in current + (pending if in_progress else []) if e.get("entity_name")}
    if observed and observed != {want["entity_name"]}:
        raise ValueError(f"Refusing to replace {endpoint_name}: it serves {sorted(observed)}.")
    if in_progress:
        if len(pending) == 1 and _entity_matches(pending[0], want):
            return "already updating"
        raise RuntimeError(f"{endpoint_name} has a different update in progress; retry when it finishes.")
    if len(current) == 1 and _entity_matches(current[0], want):
        return "unchanged"
    client.serving_endpoints.update_config(name=endpoint_name, served_entities=[desired])
    return "updated"


def main(argv: list[str] | None = None) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ServedEntityInput

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True, help="Three-level Unity Catalog model name")
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--workload-type", default="GPU_SMALL",
                    help="e.g. CPU, GPU_SMALL, GPU_MEDIUM, GPU_LARGE, GPU_LARGE_RTX")
    ap.add_argument("--workload-size", default="Small")
    ap.add_argument("--no-scale-to-zero", action="store_true")
    ap.add_argument("--profile")
    args = ap.parse_args(argv)
    try:
        wtype = workload_type(args.workload_type)
    except ValueError as e:
        ap.error(str(e))

    desired = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=str(args.model_version),
        workload_type=wtype,
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
