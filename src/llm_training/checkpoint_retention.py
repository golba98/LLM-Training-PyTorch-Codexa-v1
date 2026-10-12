"""Bounded admission and dependency planning over immutable components."""

from collections import Counter
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path

from llm_architecture.component_checkpoint import (
    atomic_json, manifest_document, reader_lock_path, store_usage,
)

POLICY_FORMAT = 'codexa-retention-policy-v1'


class RetentionLimitReached(RuntimeError):
    """No further state can be safely admitted; use the reserved final slot."""


class PublishedCheckpointReclamationError(RetentionLimitReached):
    """Model publication succeeded, but rotation did not; stop without another update."""

    def __init__(self, record: dict, error: Exception):
        super().__init__('Checkpoint published; reclamation failed: ' + str(error))
        self.checkpoint_published = True
        self.checkpoint_record = record


@dataclass(frozen=True)
class RetentionPolicy:
    max_checkpoint_bytes: int
    max_snapshots: int = 6
    max_bytes: int | None = None
    continuous: bool = False
    minimum_rollback_states: int = 1
    pinned_snapshots: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.continuous) is not bool:
            raise ValueError('Continuous retention must be boolean.')
        if type(self.minimum_rollback_states) is not int or self.minimum_rollback_states < 1:
            raise ValueError('At least one previous recovery state is required.')
        if not isinstance(self.pinned_snapshots, (tuple, list)):
            raise ValueError('Pinned snapshots must be a sequence of identifiers.')
        object.__setattr__(self, 'pinned_snapshots', tuple(self.pinned_snapshots))
        if any(not isinstance(p, str) or not p.startswith('snapshots/')
               or '..' in Path(p).parts for p in self.pinned_snapshots):
            raise ValueError('Pins must identify store-relative snapshots.')
        if not self.continuous and (self.minimum_rollback_states != 1 or self.pinned_snapshots):
            raise ValueError('Extended retention requires a fresh continuous store.')
        if type(self.max_snapshots) is not int or self.max_snapshots < 2:
            raise ValueError('At least two snapshot slots are required.')
        if self.max_snapshots < self.minimum_rollback_states + 1:
            raise ValueError('Snapshot cap cannot fit minimum rollback states.')
        if type(self.max_checkpoint_bytes) is not int or self.max_checkpoint_bytes <= 0:
            raise ValueError('A positive checkpoint byte bound is required.')
        if self.max_bytes is None:
            object.__setattr__(self, 'max_bytes',
                               (self.max_snapshots + (4 if self.continuous else 3))
                               * self.max_checkpoint_bytes + 8 * 1024**2)
        if type(self.max_bytes) is not int or self.max_bytes < 4 * self.max_checkpoint_bytes + 8 * 1024**2:
            raise ValueError('Store quota cannot fit recovery plus temporary write space.')

    def document(self):
        # Preserve byte-for-byte policy identity for already-created V2 stores.
        value = dict(format=POLICY_FORMAT, max_checkpoint_bytes=self.max_checkpoint_bytes,
                     max_snapshots=self.max_snapshots, max_bytes=self.max_bytes,
                     garbage_collection_enabled=False)
        if self.continuous:
            value.update(continuous=True, minimum_rollback_states=self.minimum_rollback_states,
                         pinned_snapshots=list(self.pinned_snapshots))
        return value

    @classmethod
    def read(cls, path: Path):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get('format') != POLICY_FORMAT or value.get('garbage_collection_enabled') is not False:
            raise ValueError('Unsupported retention policy.')
        options = {k: value[k] for k in ('max_checkpoint_bytes', 'max_snapshots', 'max_bytes')}
        options.update({k: value[k] for k in ('continuous', 'minimum_rollback_states',
                                           'pinned_snapshots') if k in value})
        return cls(**options)


def simulate_retention(snapshots: dict, protected: set, *, max_snapshots: int) -> dict:
    """Pure graph simulation; values contain ordered publication and object identities."""
    if type(max_snapshots) is not int or max_snapshots < 2:
        raise ValueError('Invalid retention snapshot limit.')
    if protected - snapshots.keys():
        raise ValueError('Retention root references an unknown snapshot.')
    selected = set(protected)
    blocked = len(selected) > max_snapshots
    if not blocked:
        ordered = sorted(snapshots, key=lambda key: (snapshots[key]['order'], key), reverse=True)
        for key in ordered:
            if len(selected) >= max_snapshots:
                break
            selected.add(key)
    references = Counter(obj for key in selected for obj in set(snapshots[key]['objects']))
    all_objects = {obj for value in snapshots.values() for obj in value['objects']}
    return dict(blocked=blocked, retained=sorted(selected),
        protected=sorted(protected), retirement_candidates=sorted(snapshots.keys() - selected),
        object_reference_counts=dict(sorted(references.items())),
        hypothetical_unreferenced_objects=sorted(all_objects - references.keys()),
        garbage_collection_enabled=False, executable=False)


class RetentionController:
    """All mutations require the existing group writer lock."""

    def __init__(self, root: Path, policy: RetentionPolicy):
        self.root, self.policy = root, policy
        self.registry_path = root / 'retention-runs.json'

    def registry(self):
        value = json.loads(self.registry_path.read_text()) if self.registry_path.exists() else {}
        if not isinstance(value, dict) or any(not isinstance(v, dict)
                or type(v.get('reserved')) is not bool or not isinstance(v.get('index'), str)
                for v in value.values()):
            raise ValueError('Invalid retention run registry.')
        return value

    def reserve(self, run_id: str, index_path: Path, *, continuous_authorized: bool = False):
        registry = self.registry()
        relative = os.path.relpath(index_path, self.root)
        if run_id in registry:
            if registry[run_id]['index'] != relative:
                raise ValueError('Run registry identity drift.')
            if registry[run_id]['reserved']:
                return
        reserved = sum(bool(r['reserved']) for r in registry.values())
        count = len(list((self.root / 'snapshots').glob('*/*.ckpt.json')))
        if not continuous_authorized and count + reserved + 1 > self.policy.max_snapshots:
            raise RetentionLimitReached('No final recovery reservation available.')
        used = store_usage(self.root)['accounted_bytes']
        if used + (2 * reserved + 4) * self.policy.max_checkpoint_bytes + 8 * 1024**2 > self.policy.max_bytes:
            raise RetentionLimitReached('No final byte reservation available.')
        registry[run_id] = dict(index=relative, reserved=True)
        if self.policy.continuous:
            registry[run_id]['initialized'] = index_path.exists()
        atomic_json(self.registry_path, registry)

    def ensure_capacity(self, run_id: str, *, final: bool = False,
                        continuous_authorized: bool = False, verified_objects: set | None = None):
        registry = self.registry()
        if run_id not in registry or not registry[run_id]['reserved']:
            raise RetentionLimitReached('Run has no final recovery reservation.')
        reserved = sum(bool(r['reserved']) for r in registry.values())
        count = len(list((self.root / 'snapshots').glob('*/*.ckpt.json')))
        extra = 0 if final else 1
        if not continuous_authorized and count + reserved + extra > self.policy.max_snapshots:
            raise RetentionLimitReached('Snapshot cap reached; final recovery slot remains reserved.')
        used = store_usage(self.root)['accounted_bytes']
        other_reserved = reserved - (1 if final else 0)
        need = used + (2 * other_reserved + 2) * self.policy.max_checkpoint_bytes + 8 * 1024**2
        if need > self.policy.max_bytes:
            raise RetentionLimitReached('Byte cap reached; no further optimizer update admitted.')
        if continuous_authorized:
            plan = self.plan(verified_objects=verified_objects)
            if plan['blocked']:
                raise RetentionLimitReached('Continuous retention graph is blocked: ' + str(plan['uncertainties']))
            # Simulate both the next publication and reserved final publication.
            # Superseded previous roles disappear only after an atomic index commit.
            roles = {owner: dict(values) for owner, values in plan['role_roots'].items()}
            owned = roles.setdefault(run_id, {})
            def advance(key):
                for n in range(self.policy.minimum_rollback_states, 1, -1):
                    prior = owned.get('previous' if n == 2 else f'rollback-{n-1}')
                    if prior is not None:
                        owned[f'rollback-{n}'] = prior
                if 'latest' in owned:
                    owned['previous'] = owned['latest']
                owned['latest'] = key
            advance('@incoming')
            if not final:
                advance('@reserved-final')
            roots = set(plan['active_readers']) | set(self.policy.pinned_snapshots) | set(plan['dependency_roots'])
            roots.update(p for values in roles.values() for p in values.values())
            # Every peer still owns an independent final reservation.
            peers = sum(r['reserved'] for owner, r in registry.items() if owner != run_id)
            if len(roots) + peers > self.policy.max_snapshots:
                raise RetentionLimitReached('Protected states and final reservations exceed snapshot quota.')
            if count > self.policy.max_snapshots:
                raise RetentionLimitReached('Pending retirement must finish before another publication.')

    def finalized(self, run_id: str):
        registry = self.registry()
        registry[run_id]['reserved'] = False
        atomic_json(self.registry_path, registry)

    def plan(self, *, known_orphans: set | None = None,
             verified_objects: set | None = None) -> dict:
        """Read/verify the complete graph; corrupt/unknown dependencies block recommendations."""
        graph, sizes, protected, uncertainties, active = {}, {}, set(), [], []
        dependency_roots = set()
        verified = verified_objects if verified_objects is not None else set()
        role_roots = {}
        manifests = sorted((self.root / 'snapshots').glob('*/*.ckpt.json'))
        for path in manifests:
            key = str(path.relative_to(self.root))
            try:
                doc, _ = manifest_document(path, verified_objects=verified)
                graph[key] = dict(order=path.stat().st_mtime_ns,
                    objects=[r['path'] for r in doc['components'].values()])
                sizes.update({r['path']: r['bytes'] for r in doc['components'].values()})
                with reader_lock_path(path).open('rb') as stream:
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        active.append(key)
                        protected.add(key)
            except (OSError, ValueError, KeyError, TypeError) as error:
                uncertainties.append(f'{key}: {error}')
        registered_runs = set()
        for run_id, entry in self.registry().items():
            registered_runs.add(run_id)
            index_path = self.root / entry['index']
            try:
                if index_path.is_symlink() or index_path.parent.is_symlink():
                    raise ValueError('Run index symlink.')
                if (self.policy.continuous and not entry.get('initialized', True)
                        and entry['reserved'] and not index_path.exists()
                        and not any(p.parent.name == run_id for p in manifests)):
                    role_roots[run_id] = {}
                    continue
                if index_path.stat().st_size > 8*1024**2:
                    raise ValueError('Run index exceeds its metadata safety bound.')
                records = json.loads(index_path.read_text())
                if not isinstance(records, dict) or 'latest' not in records:
                    raise ValueError('Run index lacks its latest recovery root.')
                role_roots[run_id] = {}
                for role, record in records.items():
                    if record['backend'] != 'optimized' or record['run_id'] != run_id:
                        raise ValueError('Run reference identity mismatch.')
                    if (index_path.parent / record['store_root']).resolve() != self.root.resolve():
                        raise ValueError('Run reference store mismatch.')
                    expected = f"snapshots/{run_id}/{record['sha256']}.ckpt.json"
                    if record['path'] != expected or expected not in graph:
                        raise ValueError('Missing/unknown retained snapshot reference.')
                    protected.add(expected)
                    role_roots[run_id][role] = expected
            except (OSError, ValueError, KeyError, TypeError) as error:
                uncertainties.append(f'{run_id}: {error}')
        if any(path.parent.name not in registered_runs for path in manifests):
            uncertainties.append('Unregistered snapshot owner.')
        dependencies = self.root / 'retention-dependencies.json'
        if dependencies.exists():
            try:
                if dependencies.is_symlink():
                    raise ValueError('Dependency registry symlink.')
                document = json.loads(dependencies.read_text())
                if document.get('format') != 'codexa-retention-dependencies-v1' or not isinstance(document.get('runs'), dict):
                    raise ValueError('Invalid dependency registry.')
                for owner, pins in document['runs'].items():
                    if owner not in registered_runs or not isinstance(pins, dict):
                        raise ValueError('Unknown dependency owner.')
                    for label, snapshot in pins.items():
                        if not isinstance(label, str) or not 0 < len(label) <= 128 or snapshot not in graph:
                            raise ValueError('Unknown pinned snapshot or label.')
                        dependency_roots.add(snapshot)
                protected.update(dependency_roots)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                uncertainties.append('Dependency registry: ' + str(error))
        if set(self.policy.pinned_snapshots) - graph.keys():
            uncertainties.append('Pinned snapshot is missing or invalid.')
        protected.update(set(self.policy.pinned_snapshots) & graph.keys())
        disk_objects = {str(p.relative_to(self.root)) for p in (self.root / 'objects').glob('*/*.pt')}
        if self.policy.continuous and disk_objects - sizes.keys() - (known_orphans or set()):
            uncertainties.append('Uncertain orphan object ownership.')
        if uncertainties:
            protected.update(graph)
        result = simulate_retention(graph, protected, max_snapshots=self.policy.max_snapshots)
        result['blocked'] = result['blocked'] or bool(uncertainties)
        candidates = result['hypothetical_unreferenced_objects']
        result.update(uncertainties=uncertainties, active_readers=active,
            graph=graph, role_roots=role_roots, dependency_roots=sorted(dependency_roots),
            policy=self.policy.document(), physical=store_usage(self.root),
            hypothetical_payload_bytes=sum(sizes[o] for o in result['object_reference_counts']),
            hypothetical_unreferenced_bytes=sum(sizes[o] for o in candidates),
            orphan_objects=sorted(disk_objects - sizes.keys()),
            reservations=self.registry(), reclamation_authorized=False)
        return result
