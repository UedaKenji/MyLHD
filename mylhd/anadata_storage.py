"""Local serialization helpers for :class:`~mylhd.anadata.KaisekiData`.

This module provides a small, well-defined schema for exporting ``KaisekiData``
instances to disk (currently pickled), validating the payload, and restoring a
new ``KaisekiData`` instance without relying on the remote open-data service.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .anadata import KaisekiData

__all__ = [
    "KaisekiDataSnapshot",
    "KaisekiDataValidationError",
    "export_kaiseki_data",
    "import_kaiseki_data",
    "instantiate_from_payload",
    "validate_payload",
]


class KaisekiDataValidationError(ValueError):
    """Raised when a serialized KaisekiData payload does not satisfy the schema."""


def _as_tuple_of_str(values: Sequence[Any], label: str) -> Tuple[str, ...]:
    try:
        return tuple(str(v) for v in values)
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise KaisekiDataValidationError(f"{label} must be an iterable of strings.") from exc


def _as_tuple_of_int(values: Sequence[Any], label: str) -> Tuple[int, ...]:
    try:
        return tuple(int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise KaisekiDataValidationError(f"{label} must be an iterable of integers.") from exc


def _iter_payload_collection(
    collection: Any,
    *,
    names: Tuple[str, ...],
    label: str,
) -> Sequence[Any]:
    if isinstance(collection, Mapping):
        missing = [name for name in names if name not in collection]
        if missing:
            raise KaisekiDataValidationError(f"{label} is missing entries for: {missing}")
        return [collection[name] for name in names]

    if isinstance(collection, (list, tuple)):
        if len(collection) != len(names):
            raise KaisekiDataValidationError(f"{label} must have length {len(names)} (got {len(collection)}).")
        return list(collection)

    raise KaisekiDataValidationError(f"{label} must be a mapping or a sequence ordered as {names}.")


def _broadcast_dimension_array(
    value: Any,
    *,
    axis: int,
    dimsizes: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise KaisekiDataValidationError(f"Dimension data for '{name}' is empty.")

    target_shape = tuple(dimsizes)

    if arr.shape == target_shape:
        return np.array(arr, copy=False)

    if arr.ndim == 1 and arr.shape[0] == dimsizes[axis]:
        shape = [1] * len(dimsizes)
        shape[axis] = dimsizes[axis]
        arr = arr.reshape(shape)
        return np.broadcast_to(arr, target_shape)

    if arr.ndim == len(dimsizes) and all((arr.shape[idx] in (1, dimsizes[idx])) for idx in range(len(dimsizes))):
        if arr.shape[axis] not in (1, dimsizes[axis]):
            raise KaisekiDataValidationError(f"Dimension data for '{name}' must vary along axis {axis}.")
        return np.broadcast_to(arr, target_shape)

    if arr.size == dimsizes[axis]:
        shape = [1] * len(dimsizes)
        shape[axis] = dimsizes[axis]
        try:
            arr = arr.reshape(shape)
        except ValueError as exc:
            raise KaisekiDataValidationError(
                f"Unable to reshape dimension data for '{name}' into axis-aligned array."
            ) from exc
        return np.broadcast_to(arr, target_shape)

    raise KaisekiDataValidationError(f"Dimension data for '{name}' cannot be broadcast to dimsizes {target_shape}.")


def _broadcast_value_array(
    value: Any,
    *,
    dimsizes: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise KaisekiDataValidationError(f"Value data for '{name}' is empty.")

    target_shape = tuple(dimsizes)
    if arr.shape == target_shape:
        return np.array(arr, copy=False)

    if arr.ndim > len(dimsizes):
        raise KaisekiDataValidationError(
            f"Value data for '{name}' has too many dimensions (expected <= {len(dimsizes)})."
        )

    ndim_gap = len(dimsizes) - arr.ndim
    try:
        arr_candidate = np.reshape(arr, arr.shape + (1,) * ndim_gap)
        maybe = np.broadcast_to(arr_candidate, target_shape)
        return maybe
    except ValueError:
        try:
            arr_candidate = np.reshape(arr, (1,) * ndim_gap + arr.shape)
            maybe = np.broadcast_to(arr_candidate, target_shape)
            return maybe
        except ValueError as exc:
            raise KaisekiDataValidationError(
                f"Value data for '{name}' cannot be broadcast to dimsizes {target_shape}."
            ) from exc


def _coerce_payload_data(
    *,
    payload: Mapping[str, Any],
    dimnames: Tuple[str, ...],
    valnames: Tuple[str, ...],
    dimsizes: Tuple[int, ...],
) -> np.ndarray:
    has_combined = "data" in payload
    dim_key = "dimdata" if "dimdata" in payload else ("dim_data" if "dim_data" in payload else None)
    val_key = "valdata" if "valdata" in payload else ("val_data" if "val_data" in payload else None)
    has_split = dim_key is not None or val_key is not None

    if has_combined and has_split:
        raise KaisekiDataValidationError(
            "Payload must supply either 'data' or both 'dimdata' and 'valdata', not a mix."
        )

    if has_combined:
        raw_data = payload["data"]
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.asarray(raw_data)
        if raw_data.ndim != len(dimsizes) + 1:
            raise KaisekiDataValidationError(f"Data has {raw_data.ndim} dimensions but expected {len(dimsizes) + 1}.")

        expected_last = len(dimnames) + len(valnames)
        expected_shape = tuple(dimsizes) + (expected_last,)
        if raw_data.shape != expected_shape:
            raise KaisekiDataValidationError(
                f"Data shape {raw_data.shape} does not match dimsizes {tuple(dimsizes)} "
                f"and expected last dimension {expected_last}."
            )
        return np.array(raw_data, copy=True)

    # Expect split payload
    if dim_key is None or val_key is None:
        raise KaisekiDataValidationError("Payload must provide both 'dimdata' and 'valdata' when 'data' is omitted.")

    dim_arrays = []
    dim_sources = _iter_payload_collection(payload[dim_key], names=dimnames, label="dimdata")
    for axis, (name, source) in enumerate(zip(dimnames, dim_sources)):
        dim_arrays.append(
            _broadcast_dimension_array(
                source,
                axis=axis,
                dimsizes=dimsizes,
                name=name,
            )
        )

    val_arrays = []
    val_sources = _iter_payload_collection(payload[val_key], names=valnames, label="valdata")
    for name, source in zip(valnames, val_sources):
        val_arrays.append(
            _broadcast_value_array(
                source,
                dimsizes=dimsizes,
                name=name,
            )
        )

    stacked = [arr[..., np.newaxis] for arr in (*dim_arrays, *val_arrays)]
    return np.concatenate(stacked, axis=-1)


@dataclass(frozen=True)
class KaisekiDataSnapshot:
    """In-memory representation of a validated KaisekiData payload."""

    name: str
    shotno: int
    subno: int
    dimnames: Tuple[str, ...]
    dimunits: Tuple[str, ...]
    valnames: Tuple[str, ...]
    valunits: Tuple[str, ...]
    dimsizes: Tuple[int, ...]
    data: np.ndarray
    date: Optional[str] = None
    comment: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_payload(self) -> Dict[str, Any]:
        """Return a pickle-friendly dictionary that passes :func:`validate_payload`."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "date": self.date,
            "shotno": int(self.shotno),
            "subno": int(self.subno),
            "dimnames": list(self.dimnames),
            "dimunits": list(self.dimunits),
            "valnames": list(self.valnames),
            "valunits": list(self.valunits),
            "dimsizes": list(self.dimsizes),
            "data": np.array(self.data, copy=True),
            "comment": self.comment,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_kaiseki(
        cls,
        data: KaisekiData,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        schema_version: int = 1,
    ) -> "KaisekiDataSnapshot":
        """Build a snapshot from an in-memory :class:`KaisekiData`."""
        meta = dict(metadata or getattr(data, "metadata", {}) or {})
        return cls(
            name=str(getattr(data, "name", "")),
            shotno=int(getattr(data, "shotno", 0)),
            subno=int(getattr(data, "subno", 0)),
            dimnames=_as_tuple_of_str(getattr(data, "dimnames", ()), "dimnames"),
            dimunits=_as_tuple_of_str(getattr(data, "dimunits", ()), "dimunits"),
            valnames=_as_tuple_of_str(getattr(data, "valnames", ()), "valnames"),
            valunits=_as_tuple_of_str(getattr(data, "valunits", ()), "valunits"),
            dimsizes=_as_tuple_of_int(getattr(data, "dimsizes", ()), "dimsizes"),
            data=np.array(getattr(data, "data"), copy=True),
            date=str(getattr(data, "date", "")) if getattr(data, "date", None) is not None else None,
            comment=str(getattr(data, "comment", "")),
            metadata=meta,
            schema_version=int(schema_version),
        )


def validate_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalise a serialized KaisekiData payload."""
    required = ["name", "shotno", "subno", "dimnames", "dimunits", "valnames", "valunits", "dimsizes"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KaisekiDataValidationError(f"Payload missing required keys: {missing}")

    schema_version = int(payload.get("schema_version", 1))
    if schema_version != 1:
        raise KaisekiDataValidationError(f"Unsupported schema_version: {schema_version}")

    dimnames = _as_tuple_of_str(payload["dimnames"], "dimnames")
    dimunits = _as_tuple_of_str(payload["dimunits"], "dimunits")
    valnames = _as_tuple_of_str(payload["valnames"], "valnames")
    valunits = _as_tuple_of_str(payload["valunits"], "valunits")
    dimsizes = _as_tuple_of_int(payload["dimsizes"], "dimsizes")

    if not (len(dimnames) == len(dimunits) == len(dimsizes)):
        raise KaisekiDataValidationError(
            "Lengths of dimnames, dimunits, and dimsizes must match "
            f"(got {len(dimnames)}, {len(dimunits)}, {len(dimsizes)})."
        )

    if len(valnames) != len(valunits):
        raise KaisekiDataValidationError(
            "Lengths of valnames and valunits must match " f"(got {len(valnames)} and {len(valunits)})."
        )

    final_data = _coerce_payload_data(
        payload=payload,
        dimnames=dimnames,
        valnames=valnames,
        dimsizes=dimsizes,
    )

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise KaisekiDataValidationError("metadata must be a mapping.")

    comment = payload.get("comment", "")
    date = payload.get("date")

    return {
        "schema_version": schema_version,
        "name": str(payload["name"]),
        "date": None if date is None else str(date),
        "shotno": int(payload["shotno"]),
        "subno": int(payload["subno"]),
        "dimnames": dimnames,
        "dimunits": dimunits,
        "valnames": valnames,
        "valunits": valunits,
        "dimsizes": dimsizes,
        "data": final_data,
        "comment": str(comment),
        "metadata": dict(metadata),
    }


def instantiate_from_payload(
    payload: Mapping[str, Any],
    *,
    cls: Type[KaisekiData] = KaisekiData,
) -> KaisekiData:
    """Return a new ``KaisekiData`` instance from a validated payload."""
    validated = validate_payload(payload)

    obj = cls.__new__(cls)  # type: ignore[misc]
    obj.name = validated["name"]
    obj.date = validated["date"]
    obj.shotno = validated["shotno"]
    obj.subno = validated["subno"]
    obj.dimnames = list(validated["dimnames"])
    obj.dimunits = list(validated["dimunits"])
    obj.valnames = list(validated["valnames"])
    obj.valunits = list(validated["valunits"])
    obj.dimsizes = list(validated["dimsizes"])
    obj.dimno = len(obj.dimnames)
    obj.valno = len(obj.valnames)
    obj.comment = validated["comment"]
    obj.data = np.array(validated["data"], copy=True)
    obj.metadata = validated["metadata"]
    obj.schema_version = validated["schema_version"]

    obj._init()
    return obj


def _dump_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    payload = dict(payload)
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_payload(path: Path) -> MutableMapping[str, Any]:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, MutableMapping):
        raise KaisekiDataValidationError("Serialized file does not contain a mapping payload.")
    return data


def export_kaiseki_data(
    data: KaisekiData,
    path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Serialize a :class:`KaisekiData` instance to ``path`` using pickle.

    Parameters
    ----------
    data:
        The source :class:`KaisekiData` instance.
    path:
        Output file path. Parent directories are created automatically.
    metadata:
        Optional extra metadata to persist alongside the numerical payload.
    overwrite:
        When ``False`` (default), :class:`FileExistsError` is raised if ``path`` exists.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set overwrite=True to replace it.")

    snapshot = KaisekiDataSnapshot.from_kaiseki(data, metadata=metadata)
    payload = snapshot.to_payload()
    return _dump_payload(output_path, payload)


def import_kaiseki_data(path: Union[str, Path], *, cls: Type[KaisekiData] = KaisekiData) -> KaisekiData:
    """
    Load a pickled :class:`KaisekiData` payload from ``path`` and instantiate it.
    """
    payload = _load_payload(Path(path))
    return instantiate_from_payload(payload, cls=cls)
