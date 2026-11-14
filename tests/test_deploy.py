from types import SimpleNamespace

import pytest
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import ServedEntityInput

from nemotron_dbx.lifecycle import deploy


class FakeClient:
    def __init__(self, endpoint: dict | None):
        self.endpoint, self.calls = endpoint, []
        self.api_client = SimpleNamespace(do=self._do)
        self.serving_endpoints = SimpleNamespace(
            create=lambda **kw: self.calls.append(("create", kw)),
            update_config=lambda **kw: self.calls.append(("update", kw)),
        )

    def _do(self, method, path):
        if self.endpoint is None:
            raise ResourceDoesNotExist("missing")
        return self.endpoint


def _desired(wtype="GPU_LARGE_RTX", version="2"):
    return ServedEntityInput(entity_name="c.s.m", entity_version=version, workload_type=deploy.workload_type(wtype),
                             workload_size="Small", scale_to_zero_enabled=True)


def _served(wtype="GPU_LARGE_RTX", version="2"):
    return {"served_entities": [{"entity_name": "c.s.m", "entity_version": version, "workload_type": wtype,
                                 "workload_size": "Small", "scale_to_zero_enabled": True}]}


def test_extra_workload_type_is_idempotent():
    # The SDK reads GPU_LARGE_RTX back as None; the raw JSON comparison still recognises our own config.
    assert deploy.submit(FakeClient({"config": _served()}), "e", _desired()) == "unchanged"
    pending = {"state": {"config_update": "IN_PROGRESS"}, "pending_config": _served()}
    assert deploy.submit(FakeClient(pending), "e", _desired()) == "already updating"


def test_submit_create_update_and_refusals():
    client = FakeClient(None)
    assert deploy.submit(client, "e", _desired()) == "created" and client.calls[0][0] == "create"
    client = FakeClient({"config": _served(version="1")})
    assert deploy.submit(client, "e", _desired()) == "updated"
    other = {"state": {"config_update": "IN_PROGRESS"}, "pending_config": _served(wtype="GPU_LARGE")}
    with pytest.raises(RuntimeError):
        deploy.submit(FakeClient(other), "e", _desired())
    foreign = {"config": {"served_entities": [{"entity_name": "other.model", "entity_version": "1"}]}}
    with pytest.raises(ValueError):
        deploy.submit(FakeClient(foreign), "e", _desired())


def test_workload_type_validation():
    assert deploy.workload_type("GPU_LARGE").value == "GPU_LARGE"
    assert deploy.workload_type("GPU_LARGE_RTX").value == "GPU_LARGE_RTX"
    with pytest.raises(ValueError, match="Unknown workload type"):
        deploy.workload_type("GPU_MEDUIM")
