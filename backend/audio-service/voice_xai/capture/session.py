"""Request-scoped PyTorch forward-hook capture implementation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.voice_xai.capture.contracts import (
    CaptureTarget,
    CapturedRepresentation,
    ExtractionBundle,
    ExtractionInterfaceError,
    Normalization,
)
from app.voice_xai.semantic.features import FeatureExtractionResult

try:  # The project does not install PyTorch until real models are available.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class CaptureSession:
    """Request-local buffer populated by hooks during a classifier forward pass."""

    def __init__(self, request_id: str, *, max_total_elements: int) -> None:
        self.request_id = request_id
        self._max_total_elements = max_total_elements
        self._captured_elements = 0
        self._captures: dict[str, dict[int, CapturedRepresentation]] = {}
        self._lock = threading.Lock()
        self.acoustic_features: FeatureExtractionResult | None = None

    def attach_acoustic_features(self, result: FeatureExtractionResult) -> None:
        self.acoustic_features = result

    def add(self, target: CaptureTarget, output: Any) -> None:
        tensor = _select_tensor(output, target.output_path)
        if tensor is None:
            capture = _skipped(target, "selected_output_is_not_a_tensor")
        elif tensor.numel() > target.max_elements:
            capture = _skipped(
                target, f"capture_exceeds_max_elements:{target.max_elements}"
            )
        else:
            with self._lock:
                exceeds_total_budget = (
                    self._captured_elements + tensor.numel() > self._max_total_elements
                )
                if not exceeds_total_budget:
                    self._captured_elements += tensor.numel()
            if exceeds_total_budget:
                capture = _skipped(
                    target,
                    f"capture_exceeds_total_elements:{self._max_total_elements}",
                )
            else:
                # A detached CPU copy prevents retaining CUDA memory or gradients.
                values = (
                    tensor.detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()
                    .copy()
                )
                capture = CapturedRepresentation(
                    branch_name=target.branch_name,
                    layer_index=target.layer_index,
                    category=target.category,
                    values=_normalize(values, target.normalization),
                    normalization=target.normalization,
                    status="captured",
                )
        with self._lock:
            self._captures.setdefault(target.branch_name, {})[target.layer_index] = capture

    def build_bundle(self) -> ExtractionBundle:
        with self._lock:
            branches = {
                branch: dict(layers) for branch, layers in self._captures.items()
            }
        return ExtractionBundle(
            request_id=self.request_id,
            branches=branches,
            acoustic_features=self.acoustic_features,
            metadata={
                "capture_branch_count": len(branches),
                "captured_elements": self._captured_elements,
                "max_total_elements": self._max_total_elements,
            },
        )


class PyTorchExtractionInterface:
    """Registers hooks once; captures only inside a request context manager."""

    def __init__(self, *, max_total_elements: int = 1_000_000) -> None:
        _require_torch()
        if max_total_elements < 1:
            raise ValueError("max_total_elements must be positive.")
        self._max_total_elements = max_total_elements
        self._active_session: ContextVar[CaptureSession | None] = ContextVar(
            "voice_xai_capture_session", default=None
        )
        self._handles: list[Any] = []

    def register_branch(self, root_module: Any, targets: list[CaptureTarget]) -> None:
        """Attach planned hooks after the real model branch successfully loads."""

        for target in targets:
            try:
                module = root_module.get_submodule(target.module_path)
            except AttributeError as error:
                raise ExtractionInterfaceError(
                    f"Cannot resolve {target.branch_name}:{target.module_path}."
                ) from error
            self._handles.append(module.register_forward_hook(self._hook(target)))

    @contextmanager
    def capture(self, request_id: str) -> Iterator[CaptureSession]:
        session = CaptureSession(
            request_id, max_total_elements=self._max_total_elements
        )
        token = self._active_session.set(session)
        try:
            yield session
        finally:
            self._active_session.reset(token)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, target: CaptureTarget) -> Callable[..., None]:
        def capture_output(_module: Any, _inputs: Any, output: Any) -> None:
            session = self._active_session.get()
            if session is not None:
                # A best-effort XAI hook must never turn an otherwise valid
                # classifier prediction into a failure.
                try:
                    session.add(target, output)
                except Exception:  # noqa: BLE001 - capture must be isolated
                    logger.exception(
                        "voice_xai_capture_failed",
                        extra={
                            "branch_name": target.branch_name,
                            "layer_index": target.layer_index,
                            "module_path": target.module_path,
                        },
                    )

        return capture_output


def _require_torch() -> None:
    if torch is None:
        raise ExtractionInterfaceError(
            "PyTorch is required for forward-hook extraction. Install the same "
            "PyTorch build used by the real classifier runtime."
        )


def _select_tensor(output: Any, path: tuple[int | str, ...]) -> Any | None:
    selected = output
    try:
        for part in path:
            selected = selected[part]
    except (IndexError, KeyError, TypeError):
        return None
    return selected if torch is not None and isinstance(selected, torch.Tensor) else None


def _skipped(target: CaptureTarget, reason: str) -> CapturedRepresentation:
    return CapturedRepresentation(
        branch_name=target.branch_name,
        layer_index=target.layer_index,
        category=target.category,
        values=None,
        normalization=target.normalization,
        status="skipped",
        reason=reason,
    )


def _normalize(values: NDArray[np.float32], method: Normalization) -> NDArray[np.float32]:
    if method == "none":
        return values
    if method == "zscore":
        return ((values - values.mean()) / (values.std() + 1e-8)).astype(np.float32)
    return ((values - values.min()) / (values.max() - values.min() + 1e-8)).astype(
        np.float32
    )
