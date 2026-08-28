"""Tests for the shared Ayla HTTP request path."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

PKG_DIR = (
    Path(__file__).resolve().parents[1] / "custom_components" / "delonghi_coffeelink"
)
PKG_NAME = "delonghi_ayla_tests"

package = types.ModuleType(PKG_NAME)
package.__path__ = [str(PKG_DIR)]
sys.modules[PKG_NAME] = package


def _load(name: str):
    full_name = f"{PKG_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PKG_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


const = _load("const")
ayla_client = _load("ayla_client")


class _FakeResponse:
    """Minimal aiohttp response context manager used by _request_json."""

    def __init__(
        self,
        status: int,
        body: object,
        *,
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.content_type = content_type
        self._text = body if isinstance(body, str) else json.dumps(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def text(self) -> str:
        return self._text


class _FakeSession:
    """Return queued responses or raise queued network errors."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _authenticated_client(session):
    client = ayla_client.DelonghiAylaClient(session, "user@example.com", "secret")
    client._access_token = "token"
    client._expires_at = time.time() + 3600
    return client


def test_get_devices_uses_request_json_and_maps_response() -> None:
    session = _FakeSession(
        _FakeResponse(
            200,
            [
                {
                    "device": {
                        "dsn": "dsn-1",
                        "product_name": "Kitchen machine",
                        "oem_model": "oem",
                        "model": "model",
                        "sw_version": "1.2.3",
                        "lan_ip": "192.0.2.1",
                        "connection_status": "Online",
                        "connected_at": "2026-08-27T08:30:00Z",
                    }
                },
                {"dsn": "dsn-2"},
            ],
        )
    )
    client = _authenticated_client(session)

    devices = asyncio.run(client.async_get_devices())

    assert [(method, url) for method, url, _kwargs in session.calls] == [
        ("GET", f"{const.AYLA_EU_ADS_URL}/apiv1/devices.json")
    ]
    assert devices == [
        ayla_client.AylaDevice(
            dsn="dsn-1",
            name="Kitchen machine",
            oem_model="oem",
            model="model",
            sw_version="1.2.3",
            lan_ip="192.0.2.1",
            connection_status="Online",
            connected_at="2026-08-27T08:30:00Z",
        ),
        ayla_client.AylaDevice(
            dsn="dsn-2",
            name="dsn-2",
            oem_model="",
            model="",
            sw_version="",
            lan_ip="",
            connection_status="Unknown",
        ),
    ]


def test_get_properties_uses_request_json_and_indexes_by_name() -> None:
    temperature = {"name": "temperature", "value": 92}
    session = _FakeSession(
        _FakeResponse(
            200,
            [
                {"property": temperature},
                {"property": {"value": "missing name"}},
                {},
            ],
        )
    )
    client = _authenticated_client(session)

    properties = asyncio.run(client.async_get_properties("dsn-1"))

    assert [(method, url) for method, url, _kwargs in session.calls] == [
        (
            "GET",
            f"{const.AYLA_EU_ADS_URL}/apiv1/dsns/dsn-1/properties.json",
        )
    ]
    assert properties == {"temperature": temperature}


@pytest.mark.parametrize(
    ("method_name", "args", "success_body"),
    [
        ("async_get_devices", (), []),
        ("async_get_properties", ("dsn-1",), []),
    ],
)
def test_read_methods_retry_transient_error(
    method_name: str, args: tuple[str, ...], success_body: object
) -> None:
    session = _FakeSession(
        _FakeResponse(504, "Gateway Time-out", content_type="text/plain"),
        _FakeResponse(200, success_body),
    )
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with patch.object(ayla_client.asyncio, "sleep", sleep):
        result = asyncio.run(getattr(client, method_name)(*args))

    assert result == ([] if method_name == "async_get_devices" else {})
    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(const.CLOUD_HTTP_RETRY_BACKOFF)


@pytest.mark.parametrize("status", sorted(const.CLOUD_TRANSIENT_HTTP_CODES))
def test_request_json_retries_each_transient_status(status: int) -> None:
    session = _FakeSession(
        _FakeResponse(status, "temporary gateway failure", content_type="text/plain"),
        _FakeResponse(200, {"ok": True}),
    )
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with patch.object(ayla_client.asyncio, "sleep", sleep):
        result = asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    assert result == {"ok": True}
    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(const.CLOUD_HTTP_RETRY_BACKOFF)


def test_get_devices_raises_after_504_retries_are_exhausted() -> None:
    session = _FakeSession(
        *[
            _FakeResponse(504, "Gateway Time-out", content_type="text/plain")
            for _ in range(const.CLOUD_HTTP_RETRY_COUNT + 1)
        ]
    )
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with (
        patch.object(ayla_client.asyncio, "sleep", sleep),
        pytest.raises(ayla_client.CloudError) as exc_info,
    ):
        asyncio.run(client.async_get_devices())

    assert exc_info.value.http_status == 504
    assert "Gateway Time-out" in str(exc_info.value)
    assert len(session.calls) == const.CLOUD_HTTP_RETRY_COUNT + 1
    assert [call.args[0] for call in sleep.await_args_list] == [
        const.CLOUD_HTTP_RETRY_BACKOFF,
        const.CLOUD_HTTP_RETRY_BACKOFF * 2,
    ]


def test_request_json_does_not_retry_non_transient_status() -> None:
    session = _FakeSession(_FakeResponse(404, {"error": "not found"}))
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with (
        patch.object(ayla_client.asyncio, "sleep", sleep),
        pytest.raises(ayla_client.CloudError) as exc_info,
    ):
        asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    assert exc_info.value.http_status == 404
    assert len(session.calls) == 1
    sleep.assert_not_awaited()


def test_request_json_retries_network_error() -> None:
    session = _FakeSession(
        aiohttp.ClientConnectionError("connection lost"),
        _FakeResponse(200, {"ok": True}),
    )
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with patch.object(ayla_client.asyncio, "sleep", sleep):
        result = asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    assert result == {"ok": True}
    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(const.CLOUD_HTTP_RETRY_BACKOFF)


def test_request_json_retries_runtime_timeout() -> None:
    session = _FakeSession(
        TimeoutError("request timed out"),
        _FakeResponse(200, {"ok": True}),
    )
    client = _authenticated_client(session)
    sleep = AsyncMock()

    with patch.object(ayla_client.asyncio, "sleep", sleep):
        result = asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    assert result == {"ok": True}
    assert len(session.calls) == 2
    sleep.assert_awaited_once_with(const.CLOUD_HTTP_RETRY_BACKOFF)


def test_authentication_error_is_not_retried_or_normalized() -> None:
    session = MagicMock()
    client = _authenticated_client(session)
    client.async_ensure_auth = AsyncMock(
        side_effect=ayla_client.AuthError("invalid credentials")
    )
    sleep = AsyncMock()

    with (
        patch.object(ayla_client.asyncio, "sleep", sleep),
        pytest.raises(ayla_client.AuthError, match="invalid credentials"),
    ):
        asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    session.request.assert_not_called()
    sleep.assert_not_awaited()


def test_transient_retry_logging_stays_at_debug(caplog) -> None:
    session = _FakeSession(
        _FakeResponse(504, "Gateway Time-out", content_type="text/plain"),
        _FakeResponse(200, {"ok": True}),
    )
    client = _authenticated_client(session)

    with (
        caplog.at_level(logging.DEBUG, logger=ayla_client.__name__),
        patch.object(ayla_client.asyncio, "sleep", AsyncMock()),
    ):
        asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )

    assert any("retry 1/2" in record.getMessage() for record in caplog.records)
    assert any(
        "succeeded after 1 retry" in record.getMessage() for record in caplog.records
    )
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_request_json_wraps_invalid_json() -> None:
    session = _FakeSession(_FakeResponse(200, "not json", content_type="text/plain"))
    client = _authenticated_client(session)

    with pytest.raises(ayla_client.CloudError, match="expected JSON"):
        asyncio.run(
            client._request_json("GET", "https://example.test/resource", op="test")
        )


@pytest.mark.parametrize(
    ("method_name", "args", "response"),
    [
        ("async_get_devices", (), None),
        ("async_get_devices", (), {}),
        ("async_get_devices", (), ["invalid"]),
        ("async_get_properties", ("dsn-1",), None),
        ("async_get_properties", ("dsn-1",), {}),
        ("async_get_properties", ("dsn-1",), ["invalid"]),
        ("async_get_properties", ("dsn-1",), [{"property": "invalid"}]),
    ],
)
def test_read_methods_reject_invalid_response_shapes(
    method_name: str, args: tuple[str, ...], response: object
) -> None:
    client = _authenticated_client(MagicMock())
    client._request_json = AsyncMock(return_value=response)

    with pytest.raises(ayla_client.CloudError):
        asyncio.run(getattr(client, method_name)(*args))
