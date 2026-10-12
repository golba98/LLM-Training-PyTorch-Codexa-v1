"""Transactional recovery publication using the existing native payload format."""

import json
import os
from pathlib import Path
import time
from typing import Mapping

import torch

from llm_training.checkpointing import _atomic_torch_save, _write_checksum
from llm_architecture.checkpointing import verify_checkpoint_checksum, _validate_training_state
from llm_training.session import atomic_document


class RecoveryStore:
    """Six rotating slots; only an atomic index publishes verified generations.

    Slots named by this new run are overwritten only while unreferenced. Latest,
    previous, best and final references never expose a partial file/sidecar pair.
    No historical directory or other run is opened for writing.
    """

    def __new__(cls, directory: Path, *, backend: str = 'legacy', store_root: Path | None = None,
                reserve_bytes: int = 100 * 1024**3, retention: dict | None = None,
                reclamation_mode: str = 'disabled'):
        if backend not in ('legacy', 'optimized'):
            raise ValueError('Unknown checkpoint storage backend.')
        if backend == 'optimized':
            from .optimized_recovery import OptimizedRecoveryStore
            return OptimizedRecoveryStore(directory, store_root=store_root, reserve_bytes=reserve_bytes,
                                          retention=retention, reclamation_mode=reclamation_mode)
        if retention is not None:
            raise ValueError('Retention is available only for optimized stores.')
        if reclamation_mode != 'disabled':
            raise ValueError('Reclamation simulation requires optimized storage.')
        return super().__new__(cls)

    def __init__(self, directory: Path, *, backend: str = 'legacy', store_root: Path | None = None,
                 reserve_bytes: int = 100 * 1024**3, retention: dict | None = None,
                 reclamation_mode: str = 'disabled') -> None:
        self.directory = Path(directory)
        self.index_path = self.directory / 'recovery-index.json'

    def index(self) -> dict:
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def resolve(self, role: str = 'latest') -> Path:
        record = self.index()[role]
        if record.get('backend') == 'optimized':
            from .optimized_recovery import OptimizedRecoveryStore
            return OptimizedRecoveryStore(self.directory).resolve(role)
        path = self.directory / record['path']
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError('Recovery index escapes its owned directory.')
        if verify_checkpoint_checksum(path) != record['sha256']:
            raise ValueError('Recovery index fingerprint mismatch.')
        return path

    def save(self, payload: Mapping, *, best: bool = False, final: bool = False,
             best_validation: bool = False) -> dict:
        """Stage, deserialize and verify before committing the recovery index."""
        started = time.monotonic()
        state = _validate_training_state(payload['training_state'])
        index = self.index()
        if any(record.get('backend') == 'optimized' for record in index.values()):
            raise ValueError('Legacy writer cannot modify an optimized recovery index.')
        for role in index:
            self.resolve(role)
        if 'latest' in index:
            previous = torch.load(self.resolve(), map_location='cpu', weights_only=False, mmap=True)
            for key in ('run_id', 'run_name', 'config', 'tokenizer_sha256'):
                if previous.get(key) != payload.get(key):
                    raise ValueError(f'Recovery identity drift: {key}.')
            if previous.get('resume_context', {}).get('configuration_sha256') != payload.get('resume_context', {}).get('configuration_sha256'):
                raise ValueError('Recovery configuration fingerprint drift.')
            del previous
        occupied = {r['path'] for r in index.values()}
        slot = next((i for i in range(6) if f'slot-{i}/checkpoint.pt' not in occupied), None)
        if slot is None:
            raise RuntimeError('No unreferenced recovery slot available.')
        path = self.directory / f'slot-{slot}/checkpoint.pt'
        _atomic_torch_save(dict(payload), path)
        _write_checksum(path)
        digest = verify_checkpoint_checksum(path)
        loaded = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if loaded['training_state'] != payload['training_state'] or loaded['run_id'] != payload['run_id']:
            raise ValueError('Serialized checkpoint identity/counters failed reload verification.')
        required = ('model_state_dict', 'optimizer_state_dict', 'scheduler_state', 'random_states', 'resume_context')
        if any(key not in loaded for key in required):
            raise ValueError('Incomplete recovery payload.')
        def verify_finite(value):
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                if not torch.isfinite(value).all().item():
                    raise ValueError('Nonfinite serialized recovery tensor; checkpoint not published.')
            elif isinstance(value, Mapping):
                for child in value.values():
                    verify_finite(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    verify_finite(child)
        verify_finite(loaded['model_state_dict'])
        verify_finite(loaded['optimizer_state_dict'])
        del loaded
        record = {'path': str(path.relative_to(self.directory)), 'sha256': digest,
                  'bytes': path.stat().st_size, 'optimizer_step': state.optimizer_step,
                  'run_name': payload['run_name'], 'run_id': payload['run_id'],
                  'verified': True, 'write_verify_seconds': time.monotonic() - started}
        atomic_document(path.with_suffix('.pt.complete.json'), record)
        # Flush the slot directory before a parent index can reference it.
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        updated = dict(index)
        if 'latest' in index:
            updated['previous'] = index['latest']
        updated['latest'] = record
        if best:
            updated['best'] = record
        if best_validation:
            updated['best_validation'] = record
        if final:
            updated['final'] = record
        atomic_document(self.index_path, updated)
        return record
