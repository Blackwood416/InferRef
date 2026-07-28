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

import ctypes
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from inferref.frontend.pytorch.identity import Identity
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

    Reads the bytes directly from the tensor's memory rather than going through
    ``Tensor.numpy()``. Two reasons:

    * ``Tensor.numpy()`` crosses PyTorch's NumPy ABI bridge, which fails
      outright when the installed NumPy major version does not match the one
      PyTorch was compiled against — ``RuntimeError: Numpy is not available``.
      Pairing an older PyTorch with NumPy 2.x is a normal thing for a user to
      end up with, and it would otherwise silently cost them every payload.
    * The tracer has no other reason to depend on NumPy. Byte extraction is the
      writer's job; parsing those bytes is the reader's, and only the reader
      needs NumPy.

    ``contiguous()`` guarantees row-major packing from ``data_ptr()``, which
    already accounts for any storage offset, so ``numel * itemsize`` bytes from
    there are exactly this tensor's logical values.
    """
    flat = tensor.detach()
    if flat.device.type != "cpu":
        flat = flat.cpu()
    flat = flat.contiguous()

    nbytes = flat.numel() * flat.element_size()
    if nbytes == 0:
        return b""
    # from_address does not own the memory, so copy while `flat` is alive.
    return bytes((ctypes.c_char * nbytes).from_address(flat.data_ptr()))


@dataclass
class TensorCapture:
    """Applies the capture policy and writes ``.irtensor`` payloads."""

    root: Path
    policy: str = "metadata"
    #: Skip payloads above this many elements (0 disables the limit).
    max_elements: int = 0

    _payload_by_key: dict[tuple, str] = field(default_factory=dict)
    _bytes_written: int = 0
    #: First failure per exception kind, and how many times it happened.
    _failures: dict[str, tuple[str, int]] = field(default_factory=dict)

    def _record_failure(self, exc: Exception) -> None:
        kind = type(exc).__name__
        message, count = self._failures.get(kind, (str(exc), 0))
        self._failures[kind] = (message, count + 1)

    @property
    def failures(self) -> list[str]:
        """Human-readable warnings for every capture failure kind seen."""
        return [
            f"tensor capture failed for {count} tensor(s) with {kind}: {message}; "
            "those values were downgraded to metadata and their testcases "
            "cannot be reproduced"
            for kind, (message, count) in sorted(self._failures.items())
        ]

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
        except Exception as exc:
            # A tensor we cannot materialise (meta device, sparse, ...) still
            # deserves its metadata rather than aborting the trace (SPEC §59).
            # But record why: degrading `--capture-tensors all` to metadata
            # silently would hand back a trace that looks fine and yields
            # testcases that cannot run.
            self._record_failure(exc)
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
        # Payload identity is deliberately distinct from *value* identity: two
        # different runtime tensors holding identical bytes in identical layout
        # are two trace values sharing one file (IR §39). See
        # Identity.payload_key for the full rationale.
        key = Identity.payload_key(record, digest)
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
