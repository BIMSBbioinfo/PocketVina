"""Results storage in the PocketVina Atlas layout: PVATLAS1 pose files + Parquet tables.

Submodules: ``posefile`` (pose file format), ``tables`` (Parquet schemas), ``writer`` (results directory writer),
``results`` (reader), ``importer`` (legacy output -> atlas layout). The main names are also available here, imported
on first use.
"""
import importlib

_EXPORTS = {
    "Reader": "posefile",
    "Writer": "posefile",
    "read_table": "tables",
    "write_table": "tables",
    "ResultsDir": "writer",
    "ProteinWriter": "writer",
    "resolve_tmp_dir": "writer",
    "atomic_place": "writer",
    "protein_id_from_path": "writer",
    "Results": "results",
    "import_legacy": "importer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
