# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One app-owned, capability-bound native AOAI route for supervised original runs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import JsonValue, TypeAdapter

from pyrit.models import config_hash

if TYPE_CHECKING:
    from pathlib import Path

    from azure.core.credentials_async import AsyncTokenCredential

    from pyrit.backend.models.original_worker import CohostRelayConfig
    from pyrit.backend.services.original_worker_preflight import CohostPreflight
    from pyrit.backend.services.original_worker_state import CohostValidationState

logger = logging.getLogger(__name__)


class OriginalRelayError(ValueError):
    """A finite relay refusal, never an upstream body, private prompt or bearer token."""

    def __init__(self, *, code: str, status_code: int = 409) -> None:
        """Record only a server-owned reason and HTTP status."""
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(kw_only=True)
class _RelayJob:
    job_ref: UUID
    run_id: UUID
    capability_sha256: str
    evidence_path: Path
    accepting: bool = True
    request_count: int = 0
    qualification_count: int = 0
    observed_tokens: int = 0
    unresolved: bool = False
    usage_complete: bool = True
    records: list[dict[str, JsonValue]] = field(default_factory=list)
    tasks: set[asyncio.Task[bytes]] = field(default_factory=set)
    receipt: dict[str, JsonValue] | None = None
    inspect_request_ids: set[str] = field(default_factory=set)


class OriginalModelRelay:
    """Own attempts, receipts and actual inflight work independently of HTTP cancellation."""

    TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
    REQUEST_BODY_TIMEOUT_SECONDS = 10.0
    REQUEST_FIELDS = frozenset(
        {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "logprobs",
            "top_logprobs",
            "stream",
        }
    )

    def __init__(
        self,
        *,
        config: CohostRelayConfig,
        preflight: CohostPreflight,
        client: httpx.AsyncClient,
        credential: AsyncTokenCredential | None = None,
        validation_state: CohostValidationState | None = None,
    ) -> None:
        """Use an app-owned HTTP client; credential construction remains native and MI-only."""
        self.config = config
        self.preflight = preflight
        self._client = client
        self._credential = credential
        self.validation_state = validation_state
        self._jobs: dict[UUID, _RelayJob] = {}
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._observed_tokens = 0
        self._last_dispatch = 0.0
        self._minute_tokens: deque[tuple[float, int]] = deque()
        self._budget_path = preflight.config.jobs_root / "relay-budget.json"
        self._unresolved = False
        self._stopping = False

    async def startup_async(self) -> None:
        """Restore monotonic aggregate limits rather than resetting them after a restart."""
        if self.validation_state is not None:
            value = self.validation_state.budget
            self._request_count = value.requests
            self._observed_tokens = value.observed_tokens
            self._unresolved = value.unresolved
        elif await asyncio.to_thread(self._budget_path.exists):
            from pyrit.backend.services.original_worker_preflight import CohostPreflight

            content = await asyncio.to_thread(CohostPreflight._read_bounded, path=self._budget_path, limit=4096)
            value = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
            requests, tokens = value.get("requests"), value.get("observed_tokens")
            if (
                type(requests) is not int
                or type(tokens) is not int
                or requests < 0
                or tokens < 0
                or type(value.get("unresolved")) is not bool
            ):
                raise OriginalRelayError(code="relay_budget_invalid", status_code=503)
            self._request_count, self._observed_tokens = requests, tokens
            self._unresolved = value["unresolved"] is True

    def admit(self, *, job_ref: UUID, run_id: UUID, run_root: Path) -> str:
        """
        Issue one evaluated-role capability for one supervisor-owned control UUID.

        Returns:
            str: The ephemeral exact-job evaluated-role capability.
        """
        if (
            not self.allows_new_run()
            or job_ref in self._jobs
            or any(job.accepting or job.tasks for job in self._jobs.values())
        ):
            raise OriginalRelayError(code="relay_unavailable", status_code=503)
        capability = secrets.token_urlsafe(32)
        self._jobs[job_ref] = _RelayJob(
            job_ref=job_ref,
            run_id=run_id,
            capability_sha256=hashlib.sha256(capability.encode()).hexdigest(),
            evidence_path=run_root / "model-request-evidence.jsonl",
        )
        return capability

    def allows_new_run(self) -> bool:
        """
        Apply the persistent aggregate stop/uncertainty barrier.

        Returns:
            bool: Whether a fresh evaluated role can be admitted.
        """
        return (
            not self._stopping
            and not self._unresolved
            and (self.validation_state is None or self.validation_state.is_available())
            and self._request_count < self.config.aggregate_max_requests
            and self._observed_tokens < self.config.aggregate_max_observed_tokens
        )

    def revoke(self, *, job_ref: UUID) -> None:
        """Stop new posts immediately without claiming that owned upstream work has drained."""
        self._jobs[job_ref].accepting = False

    def authenticate(self, *, job_ref: UUID, capability: str, allow_closed: bool = False) -> _RelayJob:
        """
        Apply only the evaluated capability, never browser/Graph authority.

        Returns:
            _RelayJob: The exact authenticated owned job.
        """
        job = self._jobs.get(job_ref)
        digest = hashlib.sha256(capability.encode()).hexdigest()
        if (
            job is None
            or not hmac.compare_digest(job.capability_sha256, digest)
            or (not allow_closed and (not job.accepting or self._stopping))
        ):
            raise OriginalRelayError(code="relay_capability_invalid", status_code=403)
        return job

    def refuse(self, *, job_ref: UUID, capability: str, code: str, status_code: int) -> None:
        """Permanently close a valid role after a malformed or oversized HTTP request."""
        job = self.authenticate(job_ref=job_ref, capability=capability)
        job.accepting = False
        logger.warning("Original evaluated role permanently refused (%s).", code)
        raise OriginalRelayError(code=code, status_code=status_code)

    async def forward_async(
        self,
        *,
        job_ref: UUID,
        capability: str,
        content: bytes,
        request_kind: Literal["original", "qualification"] = "original",
        inspect_request_id: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Forward one bounded nonstreaming request without retries or provider fallback.

        Returns:
            tuple[bytes, str]: Exact successful upstream response bytes and the observed request ID.
        """
        job = self.authenticate(job_ref=job_ref, capability=capability)
        try:
            if request_kind not in ("original", "qualification"):
                raise OriginalRelayError(code="relay_request_kind_invalid", status_code=400)
            if request_kind == "original" and (
                inspect_request_id is None
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", inspect_request_id) is None
                or inspect_request_id in job.inspect_request_ids
            ):
                raise OriginalRelayError(code="relay_inspect_request_id_invalid", status_code=400)
            if request_kind == "qualification" and inspect_request_id is not None:
                raise OriginalRelayError(code="relay_qualification_request_id_invalid", status_code=400)
            payload = self._validate_request(content)
            if request_kind == "qualification":
                self._validate_qualification(job=job, payload=payload)
            async with self._lock:
                self._check_budget(job=job)
                if inspect_request_id is not None:
                    job.inspect_request_ids.add(inspect_request_id)
                task = asyncio.create_task(
                    self._dispatch_async(
                        job=job,
                        payload=payload,
                        source=content,
                        request_kind=request_kind,
                        inspect_request_id=inspect_request_id,
                    )
                )
                job.tasks.add(task)
        except OriginalRelayError:
            job.accepting = False
            logger.warning("Original evaluated role closed after a request/cap refusal.")
            raise
        task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)
        response = await asyncio.shield(task)
        request_id = str(job.records[-1]["request_id"])
        return response, request_id

    async def close_async(self, *, job_ref: UUID, capability: str | None = None) -> dict[str, JsonValue]:
        """
        Revoke before waiting and report actual drain, including unknown dispatches.

        Returns:
            dict[str, JsonValue]: Idempotent evaluated-role closure with authentic JSONL receipts.
        """
        job = (
            self.authenticate(job_ref=job_ref, capability=capability, allow_closed=True)
            if capability is not None
            else self._jobs[job_ref]
        )
        job.accepting = False
        if job.receipt is not None:
            return TypeAdapter(dict[str, JsonValue]).validate_python(job.receipt, strict=True)
        tasks = tuple(job.tasks)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self.config.request_timeout_seconds + 5)
            if pending:
                job.unresolved = True
        evidence = self._evidence_bytes(job=job)
        receipt = self.make_close_receipt(
            run_id=job.run_id,
            active_requests=len(job.tasks),
            request_count=job.request_count,
            observed_tokens=job.observed_tokens,
            upstream_drained=not job.tasks and not job.unresolved,
            usage_complete=job.usage_complete,
            evidence=evidence,
        )
        if not job.tasks:
            job.receipt = receipt
        return TypeAdapter(dict[str, JsonValue]).validate_python(receipt, strict=True)

    def receipt(self, *, job_ref: UUID) -> dict[str, JsonValue] | None:
        """
        Return only the independently retained close receipt.

        Returns:
            dict[str, JsonValue] | None: Immutable closed-role authority, when observed.
        """
        receipt = self._jobs[job_ref].receipt
        return TypeAdapter(dict[str, JsonValue]).validate_python(receipt, strict=True) if receipt is not None else None

    def has_active_work(self) -> bool:
        """
        Report live upstream tasks regardless of a disconnected caller.

        Returns:
            bool: Whether any model request still owns the HTTP/credential resources.
        """
        return any(job.tasks for job in self._jobs.values())

    async def shutdown_async(self) -> None:
        """Revoke all roles, retain/drain dispatched work, then close native resources."""
        self._stopping = True
        for job_ref in tuple(self._jobs):
            await self.close_async(job_ref=job_ref)
        if self.has_active_work():
            raise OriginalRelayError(code="relay_shutdown_not_drained", status_code=503)
        try:
            await self._client.aclose()
        finally:
            if self._credential is not None:
                await self._credential.close()

    async def _dispatch_async(
        self,
        *,
        job: _RelayJob,
        payload: dict[str, JsonValue],
        source: bytes,
        request_kind: Literal["original", "qualification"],
        inspect_request_id: str | None,
    ) -> bytes:
        request_id = str(uuid4())
        record: dict[str, JsonValue] = {
            "schema": "cohost-model-request/v1",
            "run_id": str(job.run_id),
            "role": "evaluated",
            "request_id": request_id,
            "request_bytes": len(source),
            "request_sha256": hashlib.sha256(source).hexdigest(),
            "dispatch_state": "not_dispatched",
            "request_kind": request_kind,
            "inspect_request_id": inspect_request_id,
            "request_payload_sha256": config_hash(
                {"request": TypeAdapter(dict[str, JsonValue]).validate_json(source, strict=True)}
            ),
            "upstream_payload_sha256": config_hash({"request": payload}),
            "upstream_usage_total_tokens": None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        dispatched = False
        deadline = asyncio.get_running_loop().time() + self.config.request_timeout_seconds
        try:
            self.preflight.verify_identity_environment()
            async with asyncio.timeout_at(deadline):
                token, identity = await self._model_token_async()
            async with self._lock:
                if not job.accepting:
                    raise OriginalRelayError(code="relay_closed")
                self._check_budget(job=job, own_task=True)
                delay = self.config.min_request_interval_seconds - (monotonic() - self._last_dispatch)
            if delay > 0:
                async with asyncio.timeout_at(deadline):
                    await asyncio.sleep(delay)
            async with self._lock:
                if not job.accepting:
                    raise OriginalRelayError(code="relay_closed")
                if asyncio.get_running_loop().time() >= deadline:
                    raise OriginalRelayError(code="relay_request_deadline", status_code=504)
                job.request_count += 1
                job.qualification_count += int(request_kind == "qualification")
                self._request_count += 1
                self._last_dispatch = monotonic()
                self._unresolved = True
                await self._persist_budget_async()
                dispatched = True
                record.update(dispatch_state="dispatched", selected_identity=identity)
            url = (
                f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.deployment}/chat/completions"
                f"?api-version={self.config.api_version}"
            )
            async with asyncio.timeout_at(deadline):
                async with self._client.stream(
                    "POST",
                    url,
                    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                    json=payload,
                    follow_redirects=False,
                ) as response:
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(chunk) > self.config.max_response_bytes - len(data):
                            raise OriginalRelayError(code="relay_response_too_large", status_code=502)
                        data.extend(chunk)
                    record.update(
                        http_status=response.status_code,
                        response_bytes=len(data),
                        response_sha256=hashlib.sha256(data).hexdigest(),
                    )
                    if response.status_code != 200:
                        record["dispatch_state"] = "settled_error"
                        # Error responses do not provide authoritative generation usage.
                        job.usage_complete = False
                        raise OriginalRelayError(code="relay_upstream_rejected", status_code=502)
            tokens = self._usage_tokens(bytes(data))
            parsed = TypeAdapter(dict[str, JsonValue]).validate_json(bytes(data), strict=True)
            usage = parsed["usage"]
            assert isinstance(usage, dict)
            response_id = parsed.get("id")
            if (
                not isinstance(response_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", response_id) is None
            ):
                raise OriginalRelayError(code="relay_response_id_missing", status_code=502)
            details = usage.get("prompt_tokens_details")
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            prompt = usage["prompt_tokens"]
            assert type(prompt) is int
            if type(cached) is not int or not 0 <= cached <= prompt:
                raise OriginalRelayError(code="relay_usage_invalid", status_code=502)
            record.update(
                response_id=response_id,
                response_payload_sha256=config_hash({"response": parsed}),
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                cached_prompt_tokens=cached,
            )
            async with self._lock:
                job.observed_tokens += tokens
                self._observed_tokens += tokens
                self._minute_tokens.append((monotonic(), tokens))
                self._unresolved = False
            record.update(dispatch_state="settled", upstream_usage_total_tokens=tokens)
            if (
                job.observed_tokens >= self.config.max_observed_tokens
                or self._observed_tokens >= self.config.aggregate_max_observed_tokens
            ):
                job.accepting = False
            return bytes(data)
        except asyncio.CancelledError:
            job.accepting = False
            if dispatched:
                job.unresolved = True
                job.usage_complete = False
                record["dispatch_state"] = "unresolved"
            record["error_type"] = "CancelledError"
            raise
        except (httpx.HTTPError, TimeoutError, ValueError) as error:
            job.accepting = False
            if dispatched and record["dispatch_state"] not in ("settled", "settled_error"):
                job.unresolved = True
                job.usage_complete = False
                record["dispatch_state"] = "unresolved"
            record["error_type"] = type(error).__name__
            logger.warning("Original model request failed (%s).", type(error).__name__)
            if isinstance(error, OriginalRelayError):
                raise
            raise OriginalRelayError(code="relay_upstream_unverified", status_code=502) from error
        finally:
            job.records.append(record)
            try:
                async with self._lock:
                    self._unresolved = self._unresolved or job.unresolved or not job.usage_complete
                    await asyncio.to_thread(
                        self._write_atomic, path=job.evidence_path, content=self._evidence_bytes(job=job)
                    )
                    await self._persist_budget_async()
            except (OSError, ValueError) as error:
                job.unresolved = self._unresolved = True
                job.usage_complete = False
                logger.error("Original model evidence persistence failed (%s).", type(error).__name__)
                raise OriginalRelayError(code="relay_receipt_persistence_failed", status_code=503) from error
            finally:
                current = asyncio.current_task()
                if current is not None:
                    job.tasks.discard(current)

    def _validate_request(self, content: bytes) -> dict[str, JsonValue]:
        if len(content) > self.config.max_request_bytes:
            raise OriginalRelayError(code="relay_request_too_large", status_code=413)
        try:
            payload = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
        except ValueError as error:
            raise OriginalRelayError(code="relay_request_invalid", status_code=400) from error
        if (
            set(payload) - self.REQUEST_FIELDS
            or payload.get("model", self.config.deployment) != self.config.deployment
            or payload.get("stream", False) is not False
            or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1
            or not isinstance(payload.get("messages"), list)
            or not payload["messages"]
            or ("max_tokens" in payload and "max_completion_tokens" in payload)
        ):
            raise OriginalRelayError(code="relay_request_not_admitted", status_code=400)
        key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
        limit = payload.get(key, self.config.max_completion_tokens)
        if type(limit) is not int or not 1 <= limit <= self.config.max_completion_tokens:
            raise OriginalRelayError(code="relay_completion_limit", status_code=400)
        payload[key] = limit
        payload["model"] = self.config.deployment
        return payload

    def _check_budget(self, *, job: _RelayJob, own_task: bool = False) -> None:
        while self._minute_tokens and self._minute_tokens[0][0] <= monotonic() - 60:
            self._minute_tokens.popleft()
        if (
            self._unresolved
            or job.unresolved
            or not job.usage_complete
            or len(job.tasks) > (1 if own_task else 0)
            or job.request_count >= self.config.max_requests
            or job.observed_tokens >= self.config.max_observed_tokens
            or self._request_count >= self.config.aggregate_max_requests
            or self._observed_tokens >= self.config.aggregate_max_observed_tokens
            or sum(tokens for _, tokens in self._minute_tokens) >= self.config.minute_observed_token_limit
        ):
            raise OriginalRelayError(code="relay_budget_or_inflight_limit", status_code=429)

    @staticmethod
    def _validate_qualification(*, job: _RelayJob, payload: dict[str, JsonValue]) -> None:
        if (
            job.qualification_count != 0
            or job.request_count != 0
            or payload.get("messages") != [{"role": "user", "content": "Reply only OK"}]
            or payload.get("max_tokens") != 8
            or set(payload) - {"model", "messages", "max_tokens", "stream", "n"}
        ):
            raise OriginalRelayError(code="relay_qualification_not_admitted", status_code=400)

    async def _model_token_async(self) -> tuple[str, dict[str, JsonValue]]:
        identity = self.preflight.native_identity
        if identity is not None:
            identity.before_token(scope=self.TOKEN_SCOPE, path="model_relay")
        if self._credential is None:
            if self.preflight.config.local_test:
                raise OriginalRelayError(code="local_fixture_has_no_model_credentials", status_code=503)
            from azure.identity.aio import DefaultAzureCredential

            self._credential = DefaultAzureCredential(
                require_envvar=True, managed_identity_client_id=str(self.preflight.config.managed_identity_client_id)
            )
        token = await self._credential.get_token(self.TOKEN_SCOPE)
        if identity is not None:
            return token.token, identity.observe(access_token=token, scope=self.TOKEN_SCOPE, path="model_relay")
        try:
            segment = token.token.split(".")[1]
            claims = TypeAdapter(dict[str, JsonValue]).validate_json(
                base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)), strict=True
            )
            client = claims.get("appid") or claims.get("azp")
            if (
                client != str(self.preflight.config.managed_identity_client_id)
                or claims.get("tid") != os.getenv("ENTRA_TENANT_ID")
                or str(claims.get("aud", "")).rstrip("/") != self.TOKEN_SCOPE.removesuffix("/.default")
                or not claims.get("oid")
                or not claims.get("xms_mirid")
            ):
                raise ValueError("Unexpected native model credential identity.")
            identity = {key: claims[key] for key in ("oid", "tid", "aud", "xms_mirid")}
            identity["client_id"] = client
            return token.token, identity
        except (IndexError, KeyError, ValueError) as error:
            raise OriginalRelayError(code="relay_native_identity_mismatch", status_code=503) from error

    async def _persist_budget_async(self) -> None:
        if self.validation_state is not None:
            await self.validation_state.update_budget_async(
                requests=self._request_count, observed_tokens=self._observed_tokens, unresolved=self._unresolved
            )
            return
        value = {
            "schema_version": 1,
            "requests": self._request_count,
            "observed_tokens": self._observed_tokens,
            "unresolved": self._unresolved,
        }
        await asyncio.to_thread(
            self._write_atomic,
            path=self._budget_path,
            content=json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
        )

    @staticmethod
    def _usage_tokens(content: bytes) -> int:
        value = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
        usage = value.get("usage")
        if not isinstance(usage, dict):
            raise OriginalRelayError(code="relay_usage_missing", status_code=502)
        counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(count) is not int or count < 0 for count in counts):
            raise OriginalRelayError(code="relay_usage_invalid", status_code=502)
        prompt, completion, total = counts
        assert type(prompt) is int and type(completion) is int and type(total) is int
        if total != prompt + completion:
            raise OriginalRelayError(code="relay_usage_invalid", status_code=502)
        return total

    @staticmethod
    def _evidence_bytes(*, job: _RelayJob) -> bytes:
        return b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n" for record in job.records
        )

    @staticmethod
    def _write_atomic(*, path: Path, content: bytes) -> None:
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @staticmethod
    def make_close_receipt(
        *,
        run_id: UUID,
        active_requests: int,
        request_count: int,
        observed_tokens: int,
        upstream_drained: bool,
        usage_complete: bool,
        evidence: bytes,
    ) -> dict[str, JsonValue]:
        """
        Serialize observed native role closure without inventing requests or token usage.

        Returns:
            dict[str, JsonValue]: Source-bindable evaluated-role receipt.
        """
        fields: dict[str, JsonValue] = {
            "run_id": str(run_id),
            "role": "evaluated",
            "capability_revoked": True,
            "upstream_drained": upstream_drained,
            "active_requests": active_requests,
            "request_count": request_count,
            "upstream_usage_total_tokens": observed_tokens,
            "usage_complete": usage_complete,
            "model_request_evidence_jsonl_b64": base64.b64encode(evidence).decode(),
            "model_request_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            "model_request_evidence_bytes": len(evidence),
        }
        fields["receipt_id"] = f"cohost-model-role-{run_id}"
        fields["receipt_sha256"] = config_hash(fields)
        return fields
