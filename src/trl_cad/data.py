from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from datasets import Dataset, IterableDataset, load_dataset

from .prompts import format_rl_query


HFDataset = Dataset | IterableDataset

FAST_LOCAL_JSON_MAX_BYTES = 64 * 1024 * 1024


def _as_path_list(path: str | list[str]) -> list[str]:
    return [path] if isinstance(path, str) else list(path)


def _can_use_fast_local_json_loader(path: str | list[str], streaming: bool) -> bool:
    if streaming:
        return False

    files = _as_path_list(path)
    if not files:
        return False

    total_size = 0
    for p in files:
        pp = Path(p)
        if not pp.exists() or not pp.is_file():
            return False
        if pp.suffix.lower() not in {".jsonl", ".json"}:
            return False
        total_size += pp.stat().st_size

    return total_size <= FAST_LOCAL_JSON_MAX_BYTES


def _read_local_json_records(path: str | list[str]) -> list[dict]:
    files = _as_path_list(path)
    records: list[dict] = []

    for p in files:
        pp = Path(p)
        suffix = pp.suffix.lower()
        with pp.open("r", encoding="utf-8") as f:
            if suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            else:
                obj = json.load(f)
                if isinstance(obj, list):
                    records.extend(obj)
                elif isinstance(obj, dict):
                    records.append(obj)
                else:
                    raise ValueError(f"Unsupported JSON structure in {pp}")

    return records


def _load_json_dataset(
    path: str | list[str],
    *,
    streaming: bool = False,
    cache_dir: str | None = None,
    keep_in_memory: bool = False,
) -> HFDataset:
    if _can_use_fast_local_json_loader(path, streaming=streaming):
        return Dataset.from_list(_read_local_json_records(path))

    kwargs = {
        "path": "json",
        "data_files": path,
        "split": "train",
        "streaming": streaming,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if not streaming:
        kwargs["keep_in_memory"] = keep_in_memory
    return load_dataset(**kwargs)


def _column_names(ds: HFDataset) -> list[str]:
    if ds.column_names is not None:
        return list(ds.column_names)
    if ds.features is not None:
        return list(ds.features.keys())
    return []


def _ensure_required_columns(ds: HFDataset, required: set[str], stage_name: str) -> list[str]:
    columns = _column_names(ds)
    if not required.issubset(set(columns)):
        raise ValueError(
            f"{stage_name} schema mismatch. required={sorted(required)}, actual={sorted(columns)}"
        )
    return columns


def _safe_map(
    ds: HFDataset,
    fn: Callable,
    *,
    remove_columns: list[str],
    desc: str,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
) -> HFDataset:
    map_kwargs = {
        "remove_columns": remove_columns,
        "desc": desc,
    }
    # IterableDataset (streaming) does not support num_proc / load_from_cache_file.
    # Only pass num_proc if > 1 to avoid Windows multiprocessing issues.
    if isinstance(ds, Dataset):
        if num_proc > 1:
            map_kwargs["num_proc"] = max(1, int(num_proc))
        map_kwargs["load_from_cache_file"] = load_from_cache_file
    return ds.map(fn, **map_kwargs)


def load_stage1_dataset(
    path: str | list[str],
    *,
    streaming: bool = False,
    cache_dir: str | None = None,
    keep_in_memory: bool = False,
) -> HFDataset:
    """Stage1 data format: JSONL rows with {'text': '...scad...'}"""
    ds = _load_json_dataset(
        path,
        streaming=streaming,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )
    _ensure_required_columns(ds, {"text"}, "Stage1")
    return ds


def load_stage2_dataset(
    path: str | list[str],
    *,
    streaming: bool = False,
    cache_dir: str | None = None,
    keep_in_memory: bool = False,
    cot_field: str = "cot",
    include_cot: bool = True,
    allow_cot_fallback_fields: bool = True,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
) -> HFDataset:
    """Stage2 data format.

    Required fields: {'prompt', 'scad_code'}
    Optional CoT field: default 'cot', with fallback to reasoning/thought/analysis/chain_of_thought.
    """
    ds = _load_json_dataset(
        path,
        streaming=streaming,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )
    required = {"prompt", "scad_code"}
    columns = _ensure_required_columns(ds, required, "Stage2")

    fallback_cot_fields = ["reasoning", "thought", "analysis", "chain_of_thought"]

    def _valid_text(v) -> bool:
        return v is not None and str(v).strip() != ""

    def _pick_cot(row) -> str | None:
        if cot_field in row and _valid_text(row[cot_field]):
            return str(row[cot_field])
        if allow_cot_fallback_fields:
            for f in fallback_cot_fields:
                if f in row and _valid_text(row[f]):
                    return str(row[f])
        return None

    def _map_fn(row):
        cot = _pick_cot(row)
        prompt = format_rl_query(str(row["prompt"]))
        scad = str(row["scad_code"]).strip()
        if include_cot and cot:
            completion = "<think>\n{}\n</think>\n{}".format(str(cot).strip(), scad)
        else:
            completion = scad
        return {
            "prompt": prompt,
            "completion": completion,
        }

    return _safe_map(
        ds,
        _map_fn,
        remove_columns=columns,
        desc="Formatting Stage2 prompt/scad to training text",
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )


def load_stage2_pairs(
    path: str | list[str],
    *,
    streaming: bool = False,
    cache_dir: str | None = None,
    keep_in_memory: bool = False,
) -> HFDataset:
    """Stage2 prompt/scad pair dataset for semantic contrastive training."""
    ds = _load_json_dataset(
        path,
        streaming=streaming,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )
    _ensure_required_columns(ds, {"prompt", "scad_code"}, "Stage2-pairs")
    return ds


def load_stage3_prompts(
    path: str | list[str],
    *,
    streaming: bool = False,
    cache_dir: str | None = None,
    keep_in_memory: bool = False,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
) -> HFDataset:
    """Stage3 data format: JSONL rows with {'prompt': '...'}"""
    ds = _load_json_dataset(
        path,
        streaming=streaming,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
    )
    columns = _ensure_required_columns(ds, {"prompt"}, "Stage3")

    def _map_fn(row):
        prompt = str(row["prompt"])
        query = format_rl_query(prompt)
        return {
            "prompt": query,
            "query": query,
            "raw_prompt": prompt,
        }

    return _safe_map(
        ds,
        _map_fn,
        remove_columns=columns,
        desc="Formatting Stage3 prompt to GRPO query",
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )
