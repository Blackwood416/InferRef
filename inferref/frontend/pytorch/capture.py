"""Tensor payload capture (IR §18-§20; SPEC §14, §15, §46).

Payloads are written in **canonical logical contiguous order** (IR §20), so an
engine-side reader never has to reconstruct arbitrary PyTorch storage layouts.
The original stride/offset stay in the graph metadata for layout debugging.

Payload files are content-deduplicated: identical bytes are written once and
referenced by every value that matches. Combined with value interning in
:mod:`inferref.frontend.pytorch.identity`, this keeps a trace of a real model
from storing one copy of every weight per invocation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from inferref.ir.tensor_value import CaptureInfo, TensorHash, TensorValueRecord
from inferref.tensor import codec

#: CLI-facing capture policies (SPEC §14), mapped onto per-tensor modes (IR §18).
POLICY_ALIASES = {
    "none": "none",
    "metadata": "metadata",
    "metadata-only": "metadata",
    "hash": "hash",
    "hash-only": "hash",
    "outputs": "outputs",
    "selected": "selected",
    "all": "full",
    "full": "full",
}

HASH_ALGORITHM = "sha256"
#: Hash domain — hashes over different domains are never comparable (IR §19).
HASH_DOMAIN = "logical-contiguous-bytes"


def normalise_policy(policy: str) -> str:
    try:
        return POLICY_ALIASES[policy]
    except KeyError:
        raise ValueError(
            f"unknown capture policy {policy!r}; expected one of "
            f"{sorted(set(POLICY_ALIASES))}"
        ) from None


def logical_bytes(tensor: torch.Tensor) -> bytes:
    """Return the tensor's logical values in canonical contiguous order.

    Goes through ``view(torch.uint8)`` rather than ``.numpy()`` because that is
    the only path that works for every dtype — notably ``bfloat16``, which
    ``Tensor.numpy()`` rejects outright.
    """
    flat = tensor.detach().contiguous()
    if flat.device.type != "cpu":
        flat = flat.cpu()
    return flat.view(torch.uint8).numpy().tobytes()


@dataclass
class TensorCapture:
    """Applies the capture policy and writes ``.irtensor`` payloads."""

    root: Path
    policy: str = "metadata"
    #: Skip payloads above this many elements (0 disables the limit).
    max_elements: int = 0

    _payload_by_key: dict[tuple, str] = field(default_factory=dict)
    _bytes_written: int = 0

    @property
    def tensors_dir(self) -> Path:
        return self.root / "tensors"

    def capture(
        self,
        tensor: torch.Tensor,
        record: TensorValueRecord,
        *,
        is_output: bool,
        in_scope: bool,
    ) -> CaptureInfo:
        """Produce the :class:`CaptureInfo` for ``record`` under the policy."""
        mode = self._mode_for(is_output=is_output, in_scope=in_scope)
        if mode == "none":
            return CaptureInfo(mode="none")
        if mode == "metadata":
            return CaptureInfo(mode="metadata")

        try:
            raw = logical_bytes(tensor)
        except Exception:
            # A tensor we cannot materialise (meta device, sparse, ...) still
            # deserves its metadata rather than aborting the trace (SPEC §59).
            return CaptureInfo(mode="metadata")

        digest = hashlib.new(HASH_ALGORITHM, raw).hexdigest()
        hashes = (TensorHash(algorithm=HASH_ALGORITHM, domain=HASH_DOMAIN, value=digest),)
        if mode == "hash":
            return CaptureInfo(mode="hash", hashes=hashes)

        if self.max_elements and record.logical_numel > self.max_elements:
            return CaptureInfo(mode="hash", hashes=hashes)

        payload = self._write_payload(digest, record, raw)
        return CaptureInfo(mode="full", payload=payload, hashes=hashes)

    def _mode_for(self, *, is_output: bool, in_scope: bool) -> str:
        policy = self.policy
        if policy in ("none", "metadata", "hash", "full"):
            return policy
        if policy == "outputs":
            return "full" if is_output else "metadata"
        if policy == "selected":
            return "full" if in_scope else "metadata"
        return "metadata"

    def _write_payload(self, digest: str, record: TensorValueRecord, raw: bytes) -> str:
        # The dedup key must cover everything the .irtensor header encodes, not
        # just the bytes: a tensor and a reshaped view of it have identical
        # canonical payloads but different headers, so they cannot share a file.
        key = (
            digest,
            record.dtype,
            record.shape,
            record.stride,
            record.storage_offset_elements,
        )
        existing = self._payload_by_key.get(key)
        if existing is not None:
            return existing

        relative = f"tensors/v{record.id:08d}.irtensor"
        codec.write(
            self.root / relative,
            dtype=record.dtype,
            shape=record.shape,
            stride=record.stride,
            storage_offset=record.storage_offset_elements,
            payload=raw,
        )
        self._payload_by_key[key] = relative
        self._bytes_written += len(raw)
        return relative

    @property
    def payload_bytes(self) -> int:
        """Total payload bytes actually written (after dedup)."""
        return self._bytes_written

    @property
    def unique_payloads(self) -> int:
        return len(self._payload_by_key)
