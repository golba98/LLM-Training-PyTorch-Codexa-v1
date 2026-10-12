"""Training-owned role publication over immutable experiment-group objects."""

import json
from contextlib import contextmanager
import hashlib
import warnings
import fcntl
import os
from pathlib import Path
import shutil
import time
from typing import Mapping

from llm_architecture.component_checkpoint import (
    ComponentStore, FORMAT, atomic_json, manifest_document, read_components, reader_admission,
    reader_lock_path,
    _read_components_admitted,
)
from llm_architecture.checkpointing import _validate_training_state
from .checkpoint_retention import (
    RetentionPolicy, RetentionController, RetentionLimitReached, PublishedCheckpointReclamationError,
)


class OptimizedRecoveryStore:
    """Immutable recovery with opt-in, separately granted new-store reclamation."""

    def __init__(self, directory: Path, *, store_root: Path | None = None,
                 reserve_bytes: int = 100 * 1024**3, retention: dict | None = None,
                 reclamation_mode: str = 'disabled'):
        if reclamation_mode not in ('disabled', 'simulate'):
            raise ValueError('Training config supports disabled/simulate only; execution requires separate approval.')
        self.reclamation_mode = reclamation_mode
        self.last_retention_simulation = None
        self.directory = Path(directory).absolute()
        self.index_path = self.directory / 'recovery-index.json'
        if self.directory.is_symlink() or self.index_path.is_symlink():
            raise ValueError('Recovery directory/index cannot be symlinks.')
        self.store_root = Path(store_root).absolute() if store_root else self.directory / 'objects-store'
        self.reserve_bytes = reserve_bytes
        if self.index_path.exists():
            records = self.index()
            if any(record.get('backend') != 'optimized' for record in records.values()):
                raise ValueError('Cannot migrate an existing legacy recovery index.')
            if store_root is None and records:
                self.store_root = (self.directory / next(iter(records.values()))['store_root']).resolve()
        elif self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError('Optimized recovery requires a fresh run checkpoint directory.')

        policy_path = self.store_root / 'retention-policy.json'
        self.policy = RetentionPolicy.read(policy_path) if policy_path.exists() else None
        marker_path = self.store_root / 'store.json'
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker.get('retention_policy_sha256') and self.policy is None:
                raise ValueError('Bounded store retention policy is missing.')
        requested = RetentionPolicy(**retention) if retention is not None else None
        if requested and self.policy and requested != self.policy:
            raise ValueError('Checkpoint retention policy drift.')
        if requested and self.policy is None:
            if self.store_root.exists() and any(self.store_root.iterdir()):
                raise ValueError('Retention requires a fresh store; no historical adoption.')
            self.policy = requested
        self.last_recovery_selection = None
        if reclamation_mode != 'disabled' and self.policy is None:
            raise ValueError('Simulation requires a policy-enabled optimized store.')
        self.reclamation_authorization = None
        self._admission_key = None
        self._writer_verification_cache = None

    def index(self) -> dict:
        value = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        if not isinstance(value, dict) or any(not isinstance(r, dict) for r in value.values()):
            raise ValueError('Invalid recovery index.')
        return value

    def _record_path(self, role: str = 'latest') -> Path:
        record = self.index()[role]
        root = (self.directory / record['store_root']).resolve()
        if root != self.store_root.resolve() or record.get('backend') != 'optimized':
            raise ValueError('Recovery store identity drift.')
        expected = f"snapshots/{record['run_id']}/{record['sha256']}.ckpt.json"
        if record['path'] != expected:
            raise ValueError('Invalid recovery snapshot path.')
        path = root / record['path']
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Recovery manifest escapes its owned store.')
        if path.parent.name != record['run_id']:
            raise ValueError('Recovery manifest fingerprint/identity mismatch.')
        return path

    def resolve(self, role: str = 'latest') -> Path:
        path = self._record_path(role)
        manifest_document(path)
        return path

    def load(self, role: str = 'latest', *, allow_previous: bool = False, **kwargs) -> dict:
        """Verify once; previous fallback is explicit and never republishes an index."""
        with reader_admission(self.store_root):
            return self._load_admitted(role, allow_previous=allow_previous, **kwargs)

    def _load_admitted(self, role: str, *, allow_previous: bool, **kwargs) -> dict:
        try:
            value = _read_components_admitted(self._record_path(role), **kwargs)
            self.last_recovery_selection = dict(requested=role, selected=role, fallback=False)
            return value
        except (OSError, ValueError, KeyError, TypeError) as error:
            if not allow_previous or role != 'latest':
                raise
            value = _read_components_admitted(self._record_path('previous'), **kwargs)
            self.last_recovery_selection = dict(requested=role, selected='previous',
                                               fallback=True, error=str(error))
            warnings.warn('Latest checkpoint failed verification; explicitly recovering previous.', RuntimeWarning)
            return value

    def _component_store(self):
        digest = None
        if self.policy:
            raw = (json.dumps(self.policy.document(), sort_keys=True, separators=(',', ':')) + '\n').encode()
            digest = hashlib.sha256(raw).hexdigest()
        return ComponentStore(self.store_root, reserve_bytes=self.reserve_bytes,
                              retention_policy_sha256=digest,
                              continuous_retention=bool(self.policy and self.policy.continuous))

    @contextmanager
    def _writer(self, component=None):
        """Verification reuse is confined to one exclusive writer transaction."""
        with (component or self._component_store()).writer():
            self._writer_verification_cache = set()
            try:
                yield
            finally:
                self._writer_verification_cache = None

    def authorize_reclamation(self, grant):
        """Attach separate approval; no training configuration can issue this capability."""
        from .checkpoint_reclamation import ReclamationGrant
        if not isinstance(grant, ReclamationGrant):
            raise ValueError('A separate store-scoped reclamation grant is required.')
        grant.validate(self)
        self.reclamation_authorization = grant
        self._admission_key = None

    def _authorized(self):
        if self.reclamation_authorization is None:
            return False
        self.reclamation_authorization.validate(self)
        return True

    def _initialize_policy(self):
        """Caller holds the writer lock; existing policy is never rewritten."""
        path = self.store_root / 'retention-policy.json'
        if not path.exists():
            atomic_json(path, self.policy.document(), immutable=True)
        if RetentionPolicy.read(path) != self.policy:
            raise ValueError('Checkpoint retention policy drift.')

    def _retention(self, run_id: str):
        if self.policy is None:
            return None
        self._initialize_policy()
        controller = RetentionController(self.store_root, self.policy)
        controller.reserve(run_id, self.index_path, continuous_authorized=self._authorized())
        return controller

    def _capacity_key(self, controller):
        """Metadata-only reservation cache; never caches tensor integrity for loading."""
        registry = controller.registry()
        active = []
        for path in sorted((self.store_root / 'snapshots').glob('*/*.ckpt.json')):
            with reader_lock_path(path).open('rb') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    active.append(str(path.relative_to(self.store_root)))
        def identity(path):
            stat = path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        dependencies = self.store_root / 'retention-dependencies.json'
        return (identity(controller.registry_path), tuple(active),
                identity(dependencies) if dependencies.exists() else None,
                tuple((owner, identity(self.store_root / row['index'])
                       if (self.store_root / row['index']).exists() else None)
                      for owner, row in sorted(registry.items())))

    def prepare_update(self, run_id: str):
        """Before an optimizer update, ensure capacity beyond final reservations."""
        if self.policy is None:
            return
        with self._writer():
            controller = self._retention(run_id)
            authorized = self._authorized()
            key = self._capacity_key(controller) if authorized else None
            if not authorized or key != self._admission_key:
                if authorized:
                    from .checkpoint_reclamation import rotate_locked
                    rotate_locked(self, self.reclamation_authorization)
                controller.ensure_capacity(run_id, continuous_authorized=authorized,
                                           verified_objects=self._writer_verification_cache)
                self._admission_key = self._capacity_key(controller) if authorized else None
            if authorized and shutil.disk_usage(self.store_root).free < self.reserve_bytes + 2 * self.policy.max_checkpoint_bytes:
                raise RetentionLimitReached('Actual disk capacity cannot cover next and final publication.')

    def plan_retention(self) -> dict:
        if self.policy is None:
            raise ValueError('Retention planning requires a fresh policy-enabled store.')
        with (self.store_root / '.writer.lock').open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            with reader_admission(self.store_root):
                return RetentionController(self.store_root, self.policy).plan()

    def reference(self, role: str, target_role: str = 'latest') -> dict:
        """Assign a logical role without creating another physical snapshot."""
        allowed = {'best', 'best_behavioral', 'best_validation', 'final'}
        if role not in allowed and not role.startswith(('milestone-', 'boundary-')):
            raise ValueError('Unsupported checkpoint reference role.')
        with self._writer():
            index = self.index()
            target = index[target_role]
            if self.policy and role not in allowed and role not in {
                    f"milestone-{target['optimizer_step']}", f"boundary-{target['optimizer_step']}"}:
                raise ValueError('Bounded milestone/boundary labels must match the snapshot step.')
            manifest_document(self._record_path(target_role))
            index[role] = index[target_role]
            atomic_json(self.index_path, index)
            return index[role]

    def finalize_published_checkpoint(self, expected_optimizer_step: int) -> dict:
        """Finalize an already committed update after rotation failure, without deletion."""
        with self._writer():
            index = self.index()
            record = index['latest']
            if record['optimizer_step'] != expected_optimizer_step:
                raise ValueError('Final reference must match the completed optimizer update.')
            manifest_document(self._record_path())
            index['final'] = record
            atomic_json(self.index_path, index)
            if self.policy:
                RetentionController(self.store_root, self.policy).finalized(record['run_id'])
            return record

    def pin_snapshot(self, snapshot_id: str, *, label: str) -> None:
        """Register an explicit dependency, including another registered run's snapshot.

        Pins only accumulate. No unpin/removal API is provided in this rollout.
        """
        if not self.policy or not self.policy.continuous:
            raise ValueError('Explicit dependency registration requires a new continuous store.')
        if not isinstance(label, str) or not 0 < len(label) <= 128:
            raise ValueError('Pin label must contain 1–128 characters.')
        relative = Path(snapshot_id)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Pin must identify a store-relative snapshot.')
        with self._writer():
            manifest_document(self.store_root / relative)
            owner = self.index()['latest']['run_id']
            registry = RetentionController(self.store_root, self.policy).registry()
            if owner not in registry or relative.parts[1] not in registry:
                raise ValueError('Dependency requires complete registered run ownership.')
            path = self.store_root / 'retention-dependencies.json'
            document = json.loads(path.read_text()) if path.exists() else dict(
                format='codexa-retention-dependencies-v1', runs={})
            if document.get('format') != 'codexa-retention-dependencies-v1':
                raise ValueError('Unknown dependency registry.')
            pins = document['runs'].setdefault(owner, {})
            if label in pins and pins[label] != snapshot_id:
                raise ValueError('An explicit pin cannot be silently reassigned.')
            if label not in pins and len(pins)>=self.policy.max_snapshots:
                raise ValueError('Pin label count exceeds this owner\'s bounded metadata quota.')
            pins[label] = snapshot_id
            atomic_json(path, document)
            self._admission_key = None

    def save(self, payload: Mapping, *, best: bool = False, final: bool = False,
             best_validation: bool = False, milestone: bool = False,
             boundary: bool = False) -> dict:
        started = time.monotonic()
        state = _validate_training_state(payload['training_state'])
        run_id = payload.get('run_id')
        if self.policy and (not isinstance(run_id, str) or not run_id or len(run_id) > 128
                or Path(run_id).name != run_id or run_id in ('.', '..')):
            raise ValueError('Bounded run ID must be path-safe and at most 128 characters.')
        if self.policy and (not isinstance(payload.get('run_name'), str)
                or not 0 < len(payload['run_name']) <= 128):
            raise ValueError('Bounded run name must contain 1–128 characters.')
        store = self._component_store()
        with self._writer(store):
            index = self.index()
            verified = self._writer_verification_cache
            if index:
                previous = read_components(self._record_path(), metadata_only=True, verified_objects=verified)
                for key in ('run_id', 'run_name', 'config', 'tokenizer_sha256', 'scheduler_state'):
                    if previous.get(key) != payload.get(key):
                        raise ValueError(f'Recovery identity drift: {key}.')
                if (previous.get('resume_context') or {}).get('configuration_sha256') != (payload.get('resume_context') or {}).get('configuration_sha256'):
                    raise ValueError('Recovery configuration fingerprint drift.')
                del previous
            controller = self._retention(payload['run_id'])
            authorized = self._authorized()
            if controller:
                if authorized:
                    from .checkpoint_reclamation import rotate_locked
                    rotate_locked(self, self.reclamation_authorization)
                controller.ensure_capacity(payload['run_id'], final=final,
                                           continuous_authorized=authorized,
                                           verified_objects=verified)
                if authorized:
                    store.transient_snapshot_allowance = 1
                reservations = sum(r['reserved'] for r in controller.registry().values()) - (1 if final else 0)
                store.serialization_reserve_bytes = reservations * 2 * self.policy.max_checkpoint_bytes
            path = store.snapshot(payload, verified_objects=verified)
            manifest, _ = manifest_document(path, verified_objects=verified)
            record = dict(backend='optimized', format=FORMAT,
                store_root=os.path.relpath(self.store_root, self.directory),
                path=str(path.relative_to(self.store_root)), sha256=path.name[:-10],
                bytes=sum(r['bytes'] for r in manifest['components'].values()),
                optimizer_step=state.optimizer_step, run_name=payload['run_name'],
                run_id=payload['run_id'], verified=True,
                write_verify_seconds=time.monotonic() - started)
            updated = dict(index)
            if 'latest' in index:
                if self.policy and self.policy.continuous:
                    for n in range(self.policy.minimum_rollback_states, 1, -1):
                        prior = index.get('previous' if n == 2 else f'rollback-{n-1}')
                        if prior is not None:
                            updated[f'rollback-{n}'] = prior
                updated['previous'] = index['latest']
            updated['latest'] = record
            if best:
                updated['best'] = record
                updated['best_behavioral'] = record
            if best_validation:
                updated['best_validation'] = record
                if 'best_behavioral' not in updated:
                    updated['best'] = record
            if final:
                updated['final'] = record
            if milestone:
                updated[f'milestone-{state.optimizer_step}'] = record
            if boundary:
                updated[f'boundary-{state.optimizer_step}'] = record
            if authorized:
                # Do not publish roles whose mandatory roots already exceed quota.
                plan = controller.plan(verified_objects=verified)
                roots = set(plan['active_readers']) | set(self.policy.pinned_snapshots) | set(plan['dependency_roots'])
                roots.update(record['path'] for record in updated.values())
                roots.update(p for owner, roles in plan['role_roots'].items()
                             if owner != payload['run_id'] for p in roles.values())
                if len(roots) > self.policy.max_snapshots:
                    raise RetentionLimitReached('New roles would exceed protected-state quota; index unchanged.')
            atomic_json(self.index_path, updated)
            if controller and self.policy.continuous:
                registry = controller.registry()
                registry[payload['run_id']]['initialized'] = True
                atomic_json(controller.registry_path, registry)
            if controller and final:
                controller.finalized(payload['run_id'])
            self._admission_key = None
            if self.reclamation_mode == 'simulate':
                self.last_retention_simulation = controller.plan(verified_objects=verified)
            if authorized:
                try:
                    rotate_locked(self, self.reclamation_authorization)
                except (OSError, ValueError, RetentionLimitReached) as error:
                    # Publication is durable. Do not describe a reclamation failure as
                    # a successful rotation or an unpublished model update.
                    raise PublishedCheckpointReclamationError(record, error) from error
            return record

    def retention_inventory(self) -> dict:
        """Report physical references; never prune or delete objects."""
        manifests = list((self.store_root / 'snapshots').glob('*/*.ckpt.json'))
        referenced = set()
        for path in manifests:
            manifest, _ = manifest_document(path)
            referenced.update(r['path'] for r in manifest['components'].values())
        objects = {str(p.relative_to(self.store_root)) for p in (self.store_root / 'objects').glob('*/*.pt')}
        return dict(garbage_collection_enabled=False, manifests=len(manifests),
                    referenced_objects=sorted(referenced),
                    unreferenced_objects=sorted(objects - referenced))
