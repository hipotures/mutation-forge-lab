"""Native v3 batched AST provider boundary and transport validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from mutation_forge.models import JsonValue

from .contracts import ProgramLimits, ValidatedProgram, validate_program
from .scheduler import GeneratedEntry, ProviderBatch, ProviderCall

PROVIDER_INPUT_PROFILE_ID = "native_v3_input_4ast_128k_32ktok_v1"
PROVIDER_OUTPUT_PROFILE_ID = "native_v3_output_bounded_batch_v1"
PROGRAM_BATCH_SCHEMA_VERSION = "mforge.native.program_batch.v3"


class ProviderContractError(RuntimeError):
    """The frozen provider request or response violates its transport contract."""


class BatchedProvider(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ProviderInputProfile:
    maximum_complete_parent_asts: int = 4
    maximum_parent_references_per_slot: int = 2
    maximum_request_bytes: int = 128 * 1024
    maximum_request_tokens: int = 32 * 1024
    maximum_archive_summaries: int = 16


@dataclass(frozen=True, slots=True)
class ProviderOutputProfile:
    maximum_decoded_program_bytes: int = 32 * 1024
    maximum_encoded_response_bytes: int = 320 * 1024
    maximum_programs_per_call: int = 8
    maximum_output_tokens: int = 96 * 1024


@dataclass(frozen=True, slots=True)
class ProviderSlotSpec:
    slot_id: str
    parent_program_hashes: tuple[str, ...]
    brief: str


@dataclass(frozen=True, slots=True)
class FrozenProviderRequest:
    call: ProviderCall
    request: Mapping[str, Any]
    encoded_bytes: bytes
    conservative_token_bound: int


@dataclass(frozen=True, slots=True)
class ProviderArtifact:
    call_id: str
    repair: bool
    request_bytes: int
    raw_response: str
    entries: tuple[GeneratedEntry[ValidatedProgram], ...]
    usage: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ProviderRawArtifact:
    call_id: str
    repair: bool
    raw_response: str
    usage: Mapping[str, JsonValue]


def _token_upper_bound(encoded: bytes) -> int:
    # UTF-8 bytes are a safe model-independent upper bound for BPE-like
    # tokenizers. Deployments may inject an exact counter to use more of the
    # independent 128 KiB byte allowance without weakening the 32k ceiling.
    return len(encoded)


def build_provider_request(
    *,
    call: ProviderCall,
    slots: Sequence[ProviderSlotSpec],
    parent_programs: Mapping[str, ValidatedProgram],
    archive_summary: Mapping[str, Mapping[str, JsonValue]],
    system_prompt: str,
    output_schema: Mapping[str, Any],
    contract_bundle: Mapping[str, Any] | None = None,
    request_prompt: str | None = None,
    repair_prompt: str | None = None,
    input_profile: ProviderInputProfile | None = None,
    output_profile: ProviderOutputProfile | None = None,
    token_counter: Callable[[bytes], int] | None = None,
) -> FrozenProviderRequest:
    input_limits = input_profile or ProviderInputProfile()
    output_limits = output_profile or ProviderOutputProfile()
    if tuple(slot.slot_id for slot in slots) != call.slot_ids:
        raise ProviderContractError("slot specs do not match the frozen provider call")
    if len(slots) > output_limits.maximum_programs_per_call:
        raise ProviderContractError("provider batch exceeds the output profile")
    referenced: list[str] = []
    for slot in slots:
        if len(slot.parent_program_hashes) > input_limits.maximum_parent_references_per_slot:
            raise ProviderContractError("slot has too many parent references")
        for program_hash in slot.parent_program_hashes:
            if program_hash not in parent_programs:
                raise ProviderContractError(f"unknown frozen parent {program_hash}")
            if program_hash not in referenced:
                referenced.append(program_hash)
    full_parent_hashes = tuple(referenced[: input_limits.maximum_complete_parent_asts])
    full_parents = [
        {
            "program_hash": program_hash,
            "program_ast": parent_programs[program_hash].ast,
        }
        for program_hash in full_parent_hashes
    ]
    remaining = sorted(
        (
            {
                "program_hash": program_hash,
                "summary": dict(summary),
            }
            for program_hash, summary in archive_summary.items()
            if program_hash not in full_parent_hashes
        ),
        key=lambda value: str(value["program_hash"]),
    )[: input_limits.maximum_archive_summaries]
    prompt_payload = {
        "protocol": PROVIDER_INPUT_PROFILE_ID,
        "epoch": {
            "epoch_id": call.snapshot.epoch_id,
            "epoch_number": call.snapshot.epoch_number,
            "archive_snapshot_hash": call.snapshot.archive_snapshot_hash,
            "development_manifest_hash": call.snapshot.development_manifest_hash,
            "protocol_bundle_hash": call.snapshot.protocol_bundle_hash,
        },
        "slots": [
            {
                "slot_id": slot.slot_id,
                "parent_program_hashes": list(slot.parent_program_hashes),
                "brief": slot.brief,
            }
            for slot in slots
        ],
        "complete_parent_programs": full_parents,
        "bounded_archive_summary": remaining,
        "native_v3_contract": dict(contract_bundle or {}),
        "instruction": (
            request_prompt
            or "Return one independent declarative Native v3 policy AST per slot. "
            "Do not return Python. The host owns lineage, legality, scoring, and verification."
        ),
    }
    prompt = json.dumps(
        prompt_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    request: dict[str, Any] = {
        "idempotency_key": call.call_id,
        "call_id": call.call_id,
        "slot": call.slot_ids[0],
        "prompt": prompt,
        "system_prompt": system_prompt,
        "repair_prompt": repair_prompt or "Repair every invalid Native v3 AST in the batch.",
        "output_schema": dict(output_schema),
        "maximum_output_tokens": output_limits.maximum_output_tokens,
        "maximum_request_bytes": input_limits.maximum_request_bytes,
        "maximum_encoded_response_bytes": output_limits.maximum_encoded_response_bytes,
        "provider_input_profile_id": PROVIDER_INPUT_PROFILE_ID,
        "provider_output_profile_id": PROVIDER_OUTPUT_PROFILE_ID,
    }
    encoded = json.dumps(
        request,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > input_limits.maximum_request_bytes:
        raise ProviderContractError(
            f"actually encoded provider request exceeds {input_limits.maximum_request_bytes} bytes"
        )
    token_bound = (
        token_counter(encoded) if token_counter is not None else _token_upper_bound(encoded)
    )
    if token_bound > input_limits.maximum_request_tokens:
        raise ProviderContractError(
            f"provider request exceeds {input_limits.maximum_request_tokens} tokens"
        )
    return FrozenProviderRequest(call, request, encoded, token_bound)


class NativeV3Provider:
    """Generate, independently validate, and retain every response entry."""

    def __init__(
        self,
        provider: BatchedProvider,
        *,
        request_factory: Callable[[ProviderCall], FrozenProviderRequest],
        artifact_sink: Callable[[ProviderArtifact], None] | None = None,
        raw_artifact_sink: Callable[[ProviderRawArtifact], None] | None = None,
        output_profile: ProviderOutputProfile | None = None,
        allow_one_full_batch_repair: bool = True,
    ) -> None:
        self.provider = provider
        self.request_factory = request_factory
        self.artifact_sink = artifact_sink
        self.raw_artifact_sink = raw_artifact_sink
        self.output_profile = output_profile or ProviderOutputProfile()
        self.allow_one_full_batch_repair = allow_one_full_batch_repair

    def __call__(self, call: ProviderCall) -> ProviderBatch[ValidatedProgram]:
        return self._generate(call, entry_sink=None)

    def call_streaming(
        self,
        call: ProviderCall,
        entry_sink: Callable[[GeneratedEntry[ValidatedProgram]], None],
    ) -> ProviderBatch[ValidatedProgram]:
        """Publish validated siblings while the remainder of the batch is checked."""

        return self._generate(call, entry_sink=entry_sink)

    def _generate(
        self,
        call: ProviderCall,
        *,
        entry_sink: Callable[[GeneratedEntry[ValidatedProgram]], None] | None,
    ) -> ProviderBatch[ValidatedProgram]:
        frozen = self.request_factory(call)
        response = self.provider.generate(frozen.request)
        raw, usage = self._response_data(response)
        self._retain_raw(call.call_id, raw, usage, repair=False)
        buffered: list[GeneratedEntry[ValidatedProgram]] = []
        released = False

        def initial_entry(entry: GeneratedEntry[ValidatedProgram]) -> None:
            nonlocal released
            if entry_sink is None:
                return
            if released:
                entry_sink(entry)
                return
            buffered.append(entry)
            if entry.program is not None:
                released = True
                for buffered_entry in buffered:
                    entry_sink(buffered_entry)
                buffered.clear()

        batch, _parsed_raw, _parsed_usage = self._parse(
            call,
            response,
            entry_sink=initial_entry if entry_sink is not None else None,
        )
        self._retain(frozen, raw, batch.entries, usage, repair=False)
        if any(entry.program is not None for entry in batch.entries):
            return batch
        if not self.allow_one_full_batch_repair:
            if entry_sink is not None:
                for entry in buffered:
                    entry_sink(entry)
            return batch
        diagnostics = [
            {
                "slot_id": entry.slot_id,
                "code": "invalid_program",
                "message": entry.error or "invalid program",
            }
            for entry in batch.entries
        ]
        repair_request = {
            **dict(frozen.request),
            "repair_attempt": 1,
            "max_repairs": 1,
            "repair_of_call_id": call.call_id,
        }
        repair_response = self.provider.repair(repair_request, diagnostics)
        repair_raw, repair_usage = self._response_data(repair_response)
        self._retain_raw(call.call_id, repair_raw, repair_usage, repair=True)
        repaired, _parsed_repair_raw, _parsed_repair_usage = self._parse(
            call,
            repair_response,
            entry_sink=entry_sink,
        )
        self._retain(
            frozen,
            repair_raw,
            repaired.entries,
            repair_usage,
            repair=True,
        )
        return repaired

    def parse_persisted_response(
        self,
        call: ProviderCall,
        *,
        raw_response: str,
        repair: bool,
    ) -> ProviderBatch[ValidatedProgram]:
        """Finish validation from a response durably stored before a crash."""

        response = {"response_text": raw_response, "usage": {}}
        batch, raw, usage = self._parse(call, response)
        frozen = self.request_factory(call)
        self._retain(frozen, raw, batch.entries, usage, repair=repair)
        return batch

    def repair_persisted_batch(
        self,
        call: ProviderCall,
        entries: Sequence[GeneratedEntry[ValidatedProgram]],
        entry_sink: Callable[[GeneratedEntry[ValidatedProgram]], None] | None = None,
    ) -> ProviderBatch[ValidatedProgram]:
        """Resume the one frozen repair without repeating the initial call."""

        if not self.allow_one_full_batch_repair:
            raise ProviderContractError("the frozen provider budget does not permit a repair")
        if tuple(entry.slot_id for entry in entries) != call.slot_ids:
            raise ProviderContractError("persisted batch does not match the frozen provider call")
        if any(entry.program is not None for entry in entries):
            raise ProviderContractError("a partially valid batch is not repairable")
        frozen = self.request_factory(call)
        diagnostics = [
            {
                "slot_id": entry.slot_id,
                "code": "invalid_program",
                "message": entry.error or "invalid program",
            }
            for entry in entries
        ]
        repair_request = {
            **dict(frozen.request),
            "repair_attempt": 1,
            "max_repairs": 1,
            "repair_of_call_id": call.call_id,
        }
        response = self.provider.repair(repair_request, diagnostics)
        raw, usage = self._response_data(response)
        self._retain_raw(call.call_id, raw, usage, repair=True)
        repaired, _parsed_raw, _parsed_usage = self._parse(
            call,
            response,
            entry_sink=entry_sink,
        )
        self._retain(frozen, raw, repaired.entries, usage, repair=True)
        return repaired

    @staticmethod
    def _response_data(
        response: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, JsonValue]]:
        raw_value = response.get("response_text")
        if isinstance(raw_value, str):
            raw = raw_value
        else:
            projected = response.get("response", response.get("output", response))
            raw = json.dumps(
                projected,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        usage_value = response.get("usage", {})
        usage = (
            cast(Mapping[str, JsonValue], usage_value) if isinstance(usage_value, Mapping) else {}
        )
        return raw, usage

    def _parse(
        self,
        call: ProviderCall,
        response: Mapping[str, Any],
        *,
        entry_sink: Callable[[GeneratedEntry[ValidatedProgram]], None] | None = None,
    ) -> tuple[ProviderBatch[ValidatedProgram], str, Mapping[str, JsonValue]]:
        raw, usage = self._response_data(response)
        encoded = raw.encode("utf-8")
        if len(encoded) > self.output_profile.maximum_encoded_response_bytes:
            raise ProviderContractError("provider response exceeds encoded byte limit")
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as error:
            entries: tuple[GeneratedEntry[ValidatedProgram], ...] = tuple(
                GeneratedEntry(slot_id, None, None, f"invalid batch JSON: {error.msg}")
                for slot_id in call.slot_ids
            )
            if entry_sink is not None:
                for entry in entries:
                    entry_sink(entry)
            return ProviderBatch(call.call_id, entries, len(encoded)), raw, {}
        if not isinstance(envelope, dict):
            entries = tuple(
                GeneratedEntry(slot_id, None, None, "provider batch must be an object")
                for slot_id in call.slot_ids
            )
            if entry_sink is not None:
                for entry in entries:
                    entry_sink(entry)
            return ProviderBatch(call.call_id, entries, len(encoded)), raw, {}
        raw_entries = envelope.get("programs")
        if (
            set(envelope) != {"schema_version", "programs"}
            or envelope.get("schema_version") != PROGRAM_BATCH_SCHEMA_VERSION
            or not isinstance(raw_entries, list)
            or not 1 <= len(raw_entries) <= self.output_profile.maximum_programs_per_call
        ):
            entries = tuple(
                GeneratedEntry(slot_id, None, None, "invalid provider batch envelope")
                for slot_id in call.slot_ids
            )
            if entry_sink is not None:
                for entry in entries:
                    entry_sink(entry)
            return ProviderBatch(call.call_id, entries, len(encoded)), raw, {}
        raw_by_slot: dict[str, dict[str, object]] = {}
        duplicate_slots: set[str] = set()
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {
                "slot_id",
                "program_json_raw",
                "design_summary",
            }:
                continue
            slot_id = raw_entry.get("slot_id")
            if not isinstance(slot_id, str) or slot_id not in call.slot_ids:
                continue
            if slot_id in raw_by_slot:
                duplicate_slots.add(slot_id)
                continue
            raw_by_slot[slot_id] = raw_entry

        parsed: dict[str, GeneratedEntry[ValidatedProgram]] = {}

        def publish(entry: GeneratedEntry[ValidatedProgram]) -> None:
            parsed[entry.slot_id] = entry
            if entry_sink is not None:
                entry_sink(entry)

        for slot_id in call.slot_ids:
            if slot_id in duplicate_slots:
                publish(
                    GeneratedEntry(
                        slot_id,
                        None,
                        None,
                        "duplicate slot in provider batch",
                    )
                )
                continue
            raw_entry = raw_by_slot.get(slot_id)
            if raw_entry is None:
                publish(GeneratedEntry(slot_id, None, None, "provider omitted planned slot"))
                continue
            design_summary = raw_entry.get("design_summary")
            if (
                not isinstance(design_summary, str)
                or not design_summary
                or len(design_summary) > 2048
            ):
                publish(
                    GeneratedEntry(
                        slot_id,
                        None,
                        None,
                        "design_summary violates the provider contract",
                    )
                )
                continue
            program_raw = raw_entry.get("program_json_raw")
            if not isinstance(program_raw, str):
                publish(
                    GeneratedEntry(
                        slot_id,
                        None,
                        None,
                        "program_json_raw must be a string",
                    )
                )
                continue
            validation = validate_program(
                program_raw,
                limits=ProgramLimits(
                    maximum_decoded_bytes=self.output_profile.maximum_decoded_program_bytes
                ),
            )
            if validation.program is None:
                diagnostic = "; ".join(
                    f"{item.code}@{item.path}: {item.message}" for item in validation.diagnostics
                )
                publish(GeneratedEntry(slot_id, None, None, diagnostic))
            else:
                publish(
                    GeneratedEntry(
                        slot_id,
                        validation.program,
                        validation.program.program_hash,
                    )
                )
        return (
            ProviderBatch(
                call.call_id,
                tuple(parsed[slot_id] for slot_id in call.slot_ids),
                len(encoded),
            ),
            raw,
            usage,
        )

    def _retain_raw(
        self,
        call_id: str,
        raw_response: str,
        usage: Mapping[str, JsonValue],
        *,
        repair: bool,
    ) -> None:
        if self.raw_artifact_sink is None:
            return
        self.raw_artifact_sink(
            ProviderRawArtifact(
                call_id=call_id,
                repair=repair,
                raw_response=raw_response,
                usage=usage,
            )
        )

    def _retain(
        self,
        frozen: FrozenProviderRequest,
        raw: str,
        entries: tuple[GeneratedEntry[ValidatedProgram], ...],
        usage: Mapping[str, JsonValue],
        *,
        repair: bool,
    ) -> None:
        if self.artifact_sink is None:
            return
        self.artifact_sink(
            ProviderArtifact(
                call_id=frozen.call.call_id,
                repair=repair,
                request_bytes=len(frozen.encoded_bytes),
                raw_response=raw,
                entries=entries,
                usage=usage,
            )
        )
