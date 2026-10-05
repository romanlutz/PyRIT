# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strict, opt-in cohost configuration without importing a private Task in the web process."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import UUID

from packaging.version import Version
from pydantic import JsonValue, TypeAdapter
from sqlalchemy.engine import make_url

from pyrit import _compatibility
from pyrit.backend.models.original_worker import CohostBackendConfig
from pyrit.backend.services.original_native_identity import OriginalNativeIdentity
from pyrit.models import config_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.memory import MemoryInterface
    from pyrit.setup.configuration_loader import ConfigurationLoader


class CohostPreflightError(ValueError):
    """A safe startup failure, with no connection string or private staging contents."""


class CohostPreflight:
    """Freeze the deployment's identity, source and durable-memory authority."""

    CONFIG_ENV = "PYRIT_ORIGINAL_WORKER_CONFIG"
    CONFIG_SHA_ENV = "PYRIT_ORIGINAL_WORKER_CONFIG_SHA256"
    IDENTITY_KEYS = (
        "AZURE_TOKEN_CREDENTIALS",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
        "ENTRA_TENANT_ID",
        "ENTRA_CLIENT_ID",
        "ENTRA_ALLOWED_GROUP_IDS",
        "ENTRA_ADMIN_GROUP_ID",
        "PYRIT_ALLOW_UNAUTHENTICATED_ADMIN",
        "AZURE_SQL_DB_CONNECTION_STRING",
        "AZURE_SQL_DB_CONNECTION_STRING_PROD",
        "AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL",
        "AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN",
        "PYRIT_ORIGINAL_WORKER_CONFIG",
        "PYRIT_ORIGINAL_WORKER_CONFIG_SHA256",
    )
    CONFIG_BYTE_LIMIT = 65_536
    PROBE_BYTE_LIMIT = 4096
    WORKER_PROBE = (
        "import importlib.metadata,json,sys;"
        "from pyrit._compatibility import get_compatibility_id;"
        "print(json.dumps({'python':list(sys.version_info[:2]),"
        "'inspect':importlib.metadata.version('inspect_ai'),"
        "'compatibility_id':get_compatibility_id()}))"
    )

    def __init__(self, *, config: CohostBackendConfig, environment: Mapping[str, str]) -> None:
        """Remember only required process settings; do not retain a model bearer token."""
        self.config = config
        self._environment = {key: environment.get(key, "") for key in self.IDENTITY_KEYS}
        self.native_identity: OriginalNativeIdentity | None = None

    @classmethod
    async def from_environment_async(cls) -> CohostPreflight | None:
        """
        Load a hash-bound server descriptor.

        Returns:
            CohostPreflight | None: Explicit cohost policy, or the unchanged ordinary application.
        """
        path_value = os.getenv(cls.CONFIG_ENV, "")
        expected = os.getenv(cls.CONFIG_SHA_ENV, "")
        if not path_value and not expected:
            return None
        if not path_value or len(expected) != 64:
            raise CohostPreflightError("Original worker configuration requires its exact file and SHA256.")
        path = Path(path_value)
        if not path.is_absolute():
            raise CohostPreflightError("Original worker configuration must be an absolute server-owned path.")
        content = await asyncio.to_thread(cls._read_bounded, path=path, limit=cls.CONFIG_BYTE_LIMIT)
        if hashlib.sha256(content).hexdigest() != expected:
            raise CohostPreflightError("Original worker configuration digest differs.")
        config = CohostBackendConfig.model_validate_json(content)
        if config.local_test and os.getenv("PYRIT_DEV_MODE", "").lower() != "true":
            raise CohostPreflightError("Local SQLite fixtures cannot enable a hosted original worker.")
        return cls(config=config, environment=os.environ)

    def validate_configuration(self, *, loader: ConfigurationLoader, environment: Mapping[str, str]) -> None:
        """Reject mutable initialization, identity fallback, unowned memory and model routes."""
        config = self.config
        if (
            loader.enable_live_reinitialization
            or loader.allow_custom_initializers
            or loader.initialization_scripts
            or loader.initializer_configs
            or loader.max_concurrent_scenario_runs != 1
        ):
            raise CohostPreflightError("Original preview requires immutable startup and one Scenario slot.")
        if not config.local_test and config.source.spec.harness.config_sha256 != config_hash(config.harness_metadata):
            raise CohostPreflightError("Original source and approved unchanged harness differ.")
        if _compatibility.get_compatibility_id().rsplit("+g", 1)[-1] != config.public_commit:
            raise CohostPreflightError("Original backend and qualified public wheel commits differ.")
        if config.local_test:
            if loader.memory_db_type not in ("sqlite", "in_memory"):
                raise CohostPreflightError("Local harmless fixtures require isolated local memory.")
            return
        if loader.memory_db_type != "azure_sql":
            raise CohostPreflightError("Hosted original evidence must use backend-owned Azure SQL memory.")
        if loader.env_files != [] or loader.env_akv_ref:
            raise CohostPreflightError("Hosted original startup must use explicit process settings, not env reloading.")
        self._validate_identity(environment=environment)
        self._validate_sql(environment=environment)
        self._validate_blob(environment=environment)
        self.native_identity = OriginalNativeIdentity(
            config=config,
            tenant_id=UUID(environment["ENTRA_TENANT_ID"]),
            verify_environment=self.verify_identity_environment,
        )

    def verify_identity_environment(self) -> None:
        """Refuse a changed credential chain, catalog or authorization setting before refresh/use."""
        if not self.config.local_test and any(os.getenv(key, "") != value for key, value in self._environment.items()):
            raise CohostPreflightError("Original preview identity or durable authority changed after startup.")

    async def verify_staging_async(self) -> dict[str, int]:
        """
        Check installed child metadata, source bytes and actual resources.

        Returns:
            dict[str, int]: Safe post-hydration disk/container measurements.
        """
        config = self.config
        digest = await asyncio.to_thread(self._file_digest, path=config.worker_entrypoint)
        if digest != config.worker_entrypoint_sha256:
            raise CohostPreflightError("Original worker entrypoint differs from its approved digest.")
        if not await asyncio.to_thread(config.worker_python.is_file):
            raise CohostPreflightError("The offline qualified worker interpreter is not installed.")
        if await asyncio.to_thread(config.worker_entrypoint.resolve) != config.worker_entrypoint:
            raise CohostPreflightError("The approved worker entrypoint cannot redirect through a symlink.")
        if config.source_contract is not None:
            await asyncio.to_thread(self._validate_source_contract)
        await asyncio.to_thread(config.jobs_root.mkdir, parents=True, exist_ok=True)
        probe = await self._probe_worker_async()
        if (
            probe.get("python") != list(config.worker_python_version)
            or probe.get("inspect") != config.worker_inspect_version
            or probe.get("compatibility_id") != _compatibility.get_compatibility_id()
        ):
            raise CohostPreflightError("Original worker runtime differs from its qualified separate lock/public wheel.")
        return await self.verify_resources_async()

    def _validate_source_contract(self) -> None:
        path = self.config.source_contract
        if path is None:
            raise CohostPreflightError("Original source qualification contract is absent.")
        content = self._read_bounded(path=path, limit=self.CONFIG_BYTE_LIMIT)
        if hashlib.sha256(content).hexdigest() != self.config.source.contract_sha256:
            raise CohostPreflightError("Original staged qualification contract bytes differ.")
        contract = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
        case = self.config.source.cases[0]
        if (
            contract.get("model_route") != self.config.relay.route_metadata
            or contract.get("harness_config_template") != self.config.harness_metadata
            or contract.get("harness_profile_name") != self.config.source.spec.harness.name
            or contract.get("package") != self.config.source.spec.package.model_dump(mode="json", exclude_none=True)
            or contract.get("status") != "qualified-updated-public-wheel"
            or contract.get("public_commit") != self.config.public_commit
            or contract.get("cases")
            != [
                {
                    "task_name": case.task_name,
                    "task_version": case.task_version,
                    "sample_id": case.sample_id,
                    "epoch": case.epoch,
                }
            ]
            or contract.get("score_policy")
            != {
                "task_name": case.task_name,
                "task_version": case.task_version,
                "primary_scorer": self.config.source.primary_scorer,
                "success_direction": None,
                "success_threshold": None,
            }
            or contract.get("reviewed_display_values") != sorted(self.config.source.display_values)
        ):
            raise CohostPreflightError("Original source lacks final same-wheel qualification/source/profile binding.")

    async def verify_resources_async(self) -> dict[str, int]:
        """
        Check real free bytes/current cgroup memory, not a quota estimate.

        Returns:
            dict[str, int]: Measured safe admission resources.
        """
        return await asyncio.to_thread(self._resource_snapshot)

    async def verify_schema_async(self, *, memory: MemoryInterface) -> None:
        """Fail on a non-head/mismatched schema instead of accepting AzureSQLMemory's warning."""
        if not self.config.local_test:
            from pyrit.memory import AzureSQLMemory

            if (
                not isinstance(memory, AzureSQLMemory)
                or not memory._skip_schema_migration
                or self.native_identity is None
                or memory.azure_token_observer is not self.native_identity
            ):
                raise CohostPreflightError("Original runtime must use the explicit no-DDL Azure SQL path.")
            if memory._connection_string != self._environment["AZURE_SQL_DB_CONNECTION_STRING"]:
                raise CohostPreflightError("The constructed runtime catalog differs from preflight.")
        await asyncio.to_thread(self._check_schema, memory=memory)

    async def verify_durable_identity_async(self, *, memory: MemoryInterface) -> None:
        """Observe both ordinary SQL engines and the actual result-container client before readiness."""
        if self.config.local_test:
            return
        from pyrit.memory.storage import AzureBlobStorageIO

        identity = self.native_identity
        storage = memory.results_storage_io
        if (
            identity is None
            or not isinstance(storage, AzureBlobStorageIO)
            or storage.azure_token_observer is not identity
        ):
            raise CohostPreflightError("Original standard durable clients lack their frozen identity observer.")
        async with asyncio.timeout(60):
            await memory.get_scores_async(score_ids=[str(UUID(int=0))])
            await storage.verify_container_access_async()
        identity.require_durable_paths()

    async def _probe_worker_async(self) -> dict[str, JsonValue]:
        process = await asyncio.create_subprocess_exec(
            str(self.config.worker_python),
            "-c",
            self.WORKER_PROBE,
            cwd=str(self.config.jobs_root),
            env=self.worker_environment(run_root=None),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=self.PROBE_BYTE_LIMIT,
        )
        try:
            assert process.stdout is not None
            content = await asyncio.wait_for(process.stdout.read(self.PROBE_BYTE_LIMIT + 1), timeout=30)
            await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode != 0 or len(content) > self.PROBE_BYTE_LIMIT:
                raise CohostPreflightError("Original worker metadata probe failed or exceeded its bound.")
            return TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    def worker_environment(self, *, run_root: Path | None) -> dict[str, str]:
        """
        Bind scratch roots before imports while retaining trusted preview credentials.

        Returns:
            dict[str, str]: Server-owned child environment, never a browser-selectable profile.
        """
        environment = dict(os.environ)
        environment.update(self.config.worker_environment)
        for key in self.IDENTITY_KEYS:
            if key in self.config.worker_environment and self.config.worker_environment[key] != self._environment[key]:
                raise CohostPreflightError(
                    "The trusted worker cannot switch the preview identity or durable authority."
                )
        if run_root is not None:
            environment.update(
                {
                    "HOME": str(run_root / "home"),
                    "USERPROFILE": str(run_root / "home"),
                    "APPDATA": str(run_root / "appdata"),
                    "LOCALAPPDATA": str(run_root / "localappdata"),
                    "XDG_CACHE_HOME": str(run_root / "cache"),
                    "XDG_CONFIG_HOME": str(run_root / "config"),
                    "TMP": str(run_root / "tmp"),
                    "TEMP": str(run_root / "tmp"),
                    "TMPDIR": str(run_root / "tmp"),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONNOUSERSITE": "1",
                }
            )
        return environment

    def _validate_identity(self, *, environment: Mapping[str, str]) -> None:
        if (
            environment.get("AZURE_TOKEN_CREDENTIALS") != "ManagedIdentityCredential"
            or environment.get("AZURE_CLIENT_ID") != str(self.config.managed_identity_client_id)
            or Version(version("azure-identity")) < Version("1.24.0")
        ):
            raise CohostPreflightError("Original native sync/async credentials require the exact MI-only chain.")
        for key in ("ENTRA_CLIENT_ID", "ENTRA_TENANT_ID"):
            try:
                UUID(environment.get(key, ""))
            except ValueError as error:
                raise CohostPreflightError("Original preview requires actual Entra application/tenant IDs.") from error
        groups = {value.strip() for value in environment.get("ENTRA_ALLOWED_GROUP_IDS", "").split(",") if value.strip()}
        if (
            not groups
            or not self.config.allowed_group_ids <= groups
            or environment.get("PYRIT_ALLOW_UNAUTHENTICATED_ADMIN", "").lower() == "true"
        ):
            raise CohostPreflightError("Original preview source authorization requires enabled immutable group auth.")

    def _validate_sql(self, *, environment: Mapping[str, str]) -> None:
        connection = environment.get("AZURE_SQL_DB_CONNECTION_STRING", "")
        if not connection or connection != environment.get("AZURE_SQL_DB_CONNECTION_STRING_PROD"):
            raise CohostPreflightError("Original runtime requires exact no-migration PROD/catalog equality.")
        try:
            url = make_url(connection)
            odbc = str(url.query.get("odbc_connect", ""))
            fields = {
                key.strip().lower(): value.strip().strip("{}")
                for part in odbc.split(";")
                if "=" in part
                for key, _, value in [part.partition("=")]
            }
            catalog = fields.get("database") or url.database
            if (
                url.drivername != "mssql+pyodbc"
                or url.username is not None
                or url.password is not None
                or any(key in fields for key in ("uid", "pwd", "authentication", "accesstoken"))
                or catalog != self.config.expected_database_name
                or catalog is None
                or catalog.lower() in ("master", "tempdb", "model", "msdb")
                or fields.get("driver", str(url.query.get("driver", ""))) != "ODBC Driver 18 for SQL Server"
                or fields.get("encrypt", str(url.query.get("Encrypt", ""))).lower() not in ("yes", "strict")
                or fields.get("trustservercertificate", str(url.query.get("TrustServerCertificate", ""))).lower()
                != "no"
            ):
                raise CohostPreflightError("Original SQL requires native MI, Driver18 and explicit owned catalog/TLS.")
        except (TypeError, ValueError) as error:
            if isinstance(error, CohostPreflightError):
                raise
            raise CohostPreflightError("Original SQL connection configuration is invalid.") from error

    def _validate_blob(self, *, environment: Mapping[str, str]) -> None:
        expected = self.config.result_container_url
        parsed = urlsplit(expected)
        if (
            environment.get("AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL") != expected
            or environment.get("AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN", "")
            or parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(".blob.core.windows.net")
            or not parsed.path.strip("/")
            or "/" in parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or (
                self.config.validation_scope is not None
                and self.config.validation_scope.state_container_url.rstrip("/") == expected.rstrip("/")
            )
        ):
            raise CohostPreflightError(
                "Original result storage requires its owned container/native backend MI, no SAS."
            )

    def _resource_snapshot(self) -> dict[str, int]:
        free = shutil.disk_usage(self.config.jobs_root).free
        if free < self.config.min_free_bytes:
            raise CohostPreflightError("Actual cohost disk space is below the approved post-hydration reserve.")
        snapshot = {"free_bytes": free}
        if not self.config.local_test:
            root = Path("/sys/fs/cgroup")
            try:
                current = int((root / "memory.current").read_text().strip())
                maximum = int((root / "memory.max").read_text().strip())
            except (OSError, ValueError) as error:
                raise CohostPreflightError("The hosted cohost memory cgroup cannot be measured.") from error
            if current >= min(maximum, self.config.max_memory_current_bytes):
                raise CohostPreflightError("Actual cohost cgroup memory is above the approved admission bound.")
            snapshot.update(memory_current_bytes=current, memory_limit_bytes=maximum)
        return snapshot

    @staticmethod
    def _read_bounded(*, path: Path, limit: int) -> bytes:
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
        if not content or len(content) > limit:
            raise CohostPreflightError("Original server-owned artifact is empty or exceeds its byte bound.")
        return content

    @staticmethod
    def _file_digest(*, path: Path) -> str:
        return hashlib.sha256(CohostPreflight._read_bounded(path=path, limit=1024 * 1024)).hexdigest()

    @staticmethod
    def _check_schema(*, memory: MemoryInterface) -> None:
        from alembic.migration import MigrationContext
        from alembic.script import ScriptDirectory

        from pyrit.memory.migration import _make_config, check_schema_migrations

        engine = memory.engine
        if engine is None:
            raise CohostPreflightError("Original memory engine is not initialized.")
        with engine.connect() as connection:
            actual = set(
                MigrationContext.configure(
                    connection=connection, opts={"version_table": "pyrit_memory_alembic_version"}
                ).get_current_heads()
            )
            expected = set(ScriptDirectory.from_config(_make_config(connection=connection)).get_heads())
            if len(expected) != 1 or actual != expected:
                raise CohostPreflightError("Original runtime schema is not at the exact combined public head.")
        check_schema_migrations(engine=engine, silent=True)
