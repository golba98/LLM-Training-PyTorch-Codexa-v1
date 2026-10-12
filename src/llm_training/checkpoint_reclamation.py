"""Store-scoped, journaled reclamation of newly enrolled component stores.

Local trusted operators are the authorization boundary, not adversarial Python
callers. Training configuration cannot mint a grant. Only the fixture factory
issues grants here; production issuance is deliberately not provided.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid

from llm_architecture.component_checkpoint import (
    LIFECYCLE, atomic_json, digest_file, reader_admission, reader_lock_path,
    store_usage, store_entries, sync_directory,
)
from .checkpoint_retention import RetentionController, RetentionLimitReached

PLAN_FORMAT = 'codexa-reclamation-plan-v1'
GRANT_FORMAT = 'codexa-reclamation-grant-v1'


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _plain(path: Path) -> Path:
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Reclamation paths cannot contain symlinks.')
    return path.resolve()


def _identity(path: Path) -> list:
    info = _plain(path).stat()
    if not path.is_file() or info.st_nlink != 1:
        raise ValueError('Unknown/non-exclusive file ownership.')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


@dataclass(frozen=True)
class ReclamationGrant:
    """Separate approval bound to a new namespace, immutable marker and policy."""

    root: str
    namespace: str
    store_id: str
    marker_sha256: str
    policy_sha256: str
    approval_id: str
    purpose: str

    def validate(self, store) -> None:
        root = _plain(store.store_root)
        namespace = _plain(Path(self.namespace))
        if (root != Path(self.root) or not root.is_relative_to(namespace)
                or self.purpose not in ('disposable_fixture', 'approved_new_store')
                or not isinstance(self.approval_id, str) or not self.approval_id):
            raise ValueError('Reclamation approval scope mismatch.')
        marker = json.loads((root / 'store.json').read_text())
        if (marker.get('lifecycle') != LIFECYCLE or marker.get('store_id') != self.store_id
                or digest_file(root / 'store.json') != self.marker_sha256
                or digest_file(root / 'retention-policy.json') != self.policy_sha256
                or not store.policy or not store.policy.continuous):
            raise ValueError('Approval is not bound to this new continuous store.')
        registry = RetentionController(root, store.policy).registry()
        for entry in registry.values():
            index = _plain(root / entry['index'])
            if not index.is_relative_to(namespace):
                raise ValueError('Registered index is outside the approved namespace.')

    def document(self) -> dict:
        return dict(format=GRANT_FORMAT, **self.__dict__)


def load_reclamation_grant(store, path: Path) -> ReclamationGrant:
    """Read separately issued approval; never issue or infer approval from config."""
    path = _plain(Path(path))
    if path.is_relative_to(store.store_root):
        raise ValueError('Approval must be external to the checkpoint store.')
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.pop('format', None) != GRANT_FORMAT:
        raise ValueError('Invalid reclamation grant.')
    grant = ReclamationGrant(**value)
    grant.validate(store)
    return grant


def create_disposable_fixture_store(parent: Path, *, max_checkpoint_bytes: int,
                                    max_snapshots: int = 6, **policy):
    """Issue fixture authority only for a namespace exclusively created by this call."""
    from .optimized_recovery import OptimizedRecoveryStore
    parent = _plain(Path(parent))
    namespace = Path(tempfile.mkdtemp(prefix='codexa-v3-fixture-', dir=parent))
    store = OptimizedRecoveryStore(namespace / 'run', store_root=namespace / 'store',
        reserve_bytes=0, retention=dict(continuous=True, max_snapshots=max_snapshots,
                                      max_checkpoint_bytes=max_checkpoint_bytes, **policy))
    with store._component_store().writer():
        store._initialize_policy()
    marker = json.loads((store.store_root / 'store.json').read_text())
    grant = ReclamationGrant(str(store.store_root), str(namespace), marker['store_id'],
        digest_file(store.store_root / 'store.json'),
        digest_file(store.store_root / 'retention-policy.json'), uuid.uuid4().hex,
        'disposable_fixture')
    atomic_json(namespace / 'reclamation-authorization.json', grant.document(), immutable=True)
    grant.validate(store)
    store.authorize_reclamation(grant)
    return store, grant


def _unknown_files(root: Path) -> list[str]:
    known = {'store.json', 'retention-policy.json', 'retention-runs.json',
             '.writer.lock', '.reader-admission.lock', 'retention-dependencies.json'}
    patterns = (r'objects/[0-9a-f]{2}/[0-9a-f]{64}\.pt',
                r'snapshots/[^/]+/[0-9a-f]{64}\.ckpt\.json',
                r'snapshots/[^/]+/\.[0-9a-f]{64}\.ckpt\.json\.reader\.lock',
                r'reclamation/[0-9a-f]{64}\.(plan|status)\.json')
    unknown = []
    _plain(root)
    prefix = str(root) + '/'
    for path, info in store_entries(root, paths=False):
        if stat.S_ISREG(info.st_mode):
            relative = path[len(prefix):]
            if (relative not in known and not any(re.fullmatch(p, relative) for p in patterns)
                    and not _owned_status_stage(root, relative)):
                unknown.append(relative)
        elif not stat.S_ISDIR(info.st_mode):
            unknown.append(path[len(prefix):])
    return sorted(unknown)


def _owned_status_stage(root: Path, relative: str) -> bool:
    """Retain interrupted control writes associated with a verified durable intent.

    Ownership requires the hashed intent and store identity, not just a 'temp' name.
    Such stages are never deleted and cannot be component/recovery roots.
    """
    match = re.fullmatch(r'reclamation/\.([0-9a-f]{64})\.status-[A-Za-z0-9_-]+', relative)
    if not match:
        return False
    intent = root/'reclamation'/(match[1]+'.plan.json')
    try:
        value=json.loads(intent.read_text())
        identity=value.pop('plan_id')
        marker=json.loads((root/'store.json').read_text())
        return (identity==match[1] and _digest(value)==identity
                and value.get('format')==PLAN_FORMAT and value.get('store_root')==str(root)
                and value.get('store_id')==marker.get('store_id'))
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return False


def _status(path: Path, value: dict) -> None:
    """Stage control metadata under its durable transaction identity."""
    atomic_json(path,value,temporary_prefix='.'+value['plan_id']+'.status-')


def _dependencies(store, inventory: dict) -> dict:
    root = store.store_root
    files = ['store.json', 'retention-policy.json', 'retention-runs.json']
    if (root/'retention-dependencies.json').exists():
        files.append('retention-dependencies.json')
    files += sorted(inventory['graph'])
    files += sorted({p for g in inventory['graph'].values() for p in g['objects']})
    files += sorted(r['index'] for r in inventory['reservations'].values()
                    if (root / r['index']).exists())
    # Hashes were checked by the inventory; identities bind the plan to that view.
    return {p: _identity(root / p) for p in files}


def _build_locked(store, *, known_orphans: set | None = None,
                  verified_objects: set | None = None) -> dict:
    controller = RetentionController(store.store_root, store.policy)
    verified = verified_objects if verified_objects is not None else store._writer_verification_cache
    inventory = controller.plan(known_orphans=known_orphans, verified_objects=verified)
    unknown = _unknown_files(store.store_root)
    inventory['uncertainties'].extend('Unknown file: ' + p for p in unknown)
    inventory['blocked'] = inventory['blocked'] or bool(unknown)
    plan = dict(format=PLAN_FORMAT, store_root=str(store.store_root),
        store_id=json.loads((store.store_root / 'store.json').read_text())['store_id'],
        dependencies=_dependencies(store, inventory), inventory=inventory,
        executable=False, authorization_required=True)
    plan['plan_id'] = _digest(plan)
    return plan


def plan_reclamation(store) -> dict:
    """Read-only planning; an immutable plan is not execution authorization."""
    if not store.policy or not store.policy.continuous:
        raise ValueError('Planning V3 reclamation requires a new continuous store.')
    with (store.store_root / '.writer.lock').open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        with reader_admission(store.store_root):
            return _build_locked(store, verified_objects=set())


def _checked_plan(store, plan: dict) -> None:
    value = dict(plan)
    identity = value.pop('plan_id', None)
    if (identity != _digest(value) or plan.get('format') != PLAN_FORMAT
            or plan.get('store_root') != str(store.store_root)
            or plan['inventory']['blocked']):
        raise ValueError('Invalid or blocked reclamation plan.')


def _remove(root: Path, relative: str, expected: list) -> bool:
    path = _plain(root / relative)
    if not path.is_relative_to(root):
        raise ValueError('Deletion target escapes its store.')
    if not path.exists():
        return False  # Intent preceded unlink: crash replay is idempotent.
    if _identity(path) != expected:
        raise ValueError('Reclamation target changed since planning.')
    path.unlink()
    sync_directory(path.parent)
    return True


def _execute_locked(store, plan: dict, grant: ReclamationGrant, *, resume: bool = False) -> dict:
    """Caller holds writer lock; gate and candidate leases remain held to completion."""
    grant.validate(store)
    _checked_plan(store, plan)
    root = store.store_root
    journal = root / 'reclamation' / (plan['plan_id'] + '.plan.json')
    status = journal.with_name(plan['plan_id'] + '.status.json')
    if resume and (not journal.exists() or json.loads(journal.read_text()) != plan):
        raise ValueError('Missing or changed durable reclamation intent.')
    inventory = plan['inventory']
    snapshots = inventory['retirement_candidates']
    objects = inventory['hypothetical_unreferenced_objects']
    with reader_admission(root, exclusive=True), ExitStack() as leases:
        current = _build_locked(store, known_orphans=set(objects) if resume else None)
        if current['inventory']['blocked']:
            raise ValueError('Current graph is uncertain; reclamation aborted.')
        if not resume and current['plan_id'] != plan['plan_id']:
            raise ValueError('Stale reclamation plan; replan before execution.')
        if resume:
            # Finalizing an already published update may add a final alias after an
            # I/O failure. Revalidated mutable roots may change; immutable retained
            # bytes cannot. Newly protected candidates are rejected below.
            mutable = {'retention-runs.json', 'retention-dependencies.json'}
            mutable.update(r['index'] for r in plan['inventory']['reservations'].values())
            for relative, identity in plan['dependencies'].items():
                if relative in snapshots or relative in objects or relative in mutable:
                    continue
                if _identity(root / relative) != identity:
                    raise ValueError('Recovery roots changed during interrupted reclamation.')
        live = set(current['inventory']['protected'])
        if live & set(snapshots):
            raise ValueError('A retirement candidate is now protected.')
        for relative in snapshots:
            path = root / relative
            lock = reader_lock_path(path)
            if lock.exists():
                stream = leases.enter_context(_plain(lock).open('rb'))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError('Active reader prevents reclamation.') from None
            elif path.exists():
                raise ValueError('Candidate reader lock is missing.')
        # Rehash candidates on resumed execution; ownership never comes from size alone.
        for relative in objects:
            path = root / relative
            cache = store._writer_verification_cache
            key = (str(root.resolve()), path.stem)
            if path.exists() and (cache is None or key not in cache) and digest_file(path) != path.stem:
                raise ValueError('Corrupt reclamation candidate.')
        lock_identities = {str(reader_lock_path(root / p).relative_to(root)):
                           _identity(reader_lock_path(root / p))
                           for p in snapshots if reader_lock_path(root / p).exists()}
        if not resume:
            # Lock identities are part of the durable transaction, not checkpoint format.
            journal_bound = 2 * len(json.dumps(plan).encode()) + 128 * 1024
            if store_usage(root)['accounted_bytes'] + journal_bound > store.policy.max_bytes:
                raise RetentionLimitReached('Insufficient accounted quota for durable reclamation journal.')
            atomic_json(journal, plan, immutable=True)
            _status(status, dict(phase='intent', lock_identities=lock_identities,
                                 removed=[], plan_id=plan['plan_id'],approval_id=grant.approval_id))
        else:
            state = json.loads(status.read_text()) if status.exists() else {}
            lock_identities = state.get('lock_identities', lock_identities)
        removed, operations = [], []
        free_before = shutil.disk_usage(root).free
        for relative in snapshots:
            changed = _remove(root, relative, plan['dependencies'][relative])
            operations.append(dict(path=relative, result='unlinked' if changed else 'already-absent'))
            removed.append(relative)
        # All remaining manifests are roots for object reachability, including unpinned ones.
        remaining = RetentionController(root, store.policy).plan(known_orphans=set(objects),
            verified_objects=store._writer_verification_cache)
        if remaining['blocked']:
            raise ValueError('Post-retirement graph is uncertain; objects retained.')
        reachable = {p for g in remaining['graph'].values() for p in g['objects']}
        if reachable & set(objects):
            raise ValueError('Object is still referenced by a remaining snapshot.')
        _status(status, dict(phase='manifests-retired', lock_identities=lock_identities,
                             removed=removed, plan_id=plan['plan_id'],approval_id=grant.approval_id))
        for relative in objects:
            changed = _remove(root, relative, plan['dependencies'][relative])
            operations.append(dict(path=relative, result='unlinked' if changed else 'already-absent'))
            removed.append(relative)
        for relative, identity in lock_identities.items():
            changed = _remove(root, relative, identity)
            operations.append(dict(path=relative, result='unlinked' if changed else 'already-absent'))
            removed.append(relative)
        final = RetentionController(root, store.policy).plan(verified_objects=store._writer_verification_cache)
        if final['blocked']:
            raise ValueError('Final graph verification failed.')
        result = dict(phase='complete', removed=removed, lock_identities=lock_identities,
                      plan_id=plan['plan_id'], physical=store_usage(root),
                      approval_id=grant.approval_id, approval_purpose=grant.purpose,
                      reclamation_executed=True, operations=operations,
                      free_before_bytes=free_before, free_after_bytes=shutil.disk_usage(root).free)
        result['retained_control_stages']=[str(p.relative_to(root)) for p in (root/'reclamation').iterdir()
                                          if p.is_file() and _owned_status_stage(root,str(p.relative_to(root)))]
        _status(status, result)
        return result


def execute_reclamation(store, plan: dict, grant: ReclamationGrant) -> dict:
    """Execute only with independent store-scoped approval, never a config boolean."""
    if not isinstance(grant, ReclamationGrant):
        raise ValueError('Explicit store-scoped reclamation grant required.')
    # Reject foreign/historical stores before even opening a writer in append mode.
    grant.validate(store)
    _checked_plan(store, plan)
    with store._writer():
        journal = store.store_root / 'reclamation' / (plan['plan_id'] + '.plan.json')
        status = journal.with_name(plan['plan_id'] + '.status.json')
        if status.exists():
            state = json.loads(status.read_text())
            if state.get('phase') == 'complete':
                grant.validate(store)
                _checked_plan(store, plan)
                return dict(state, already_complete=True)
        return _execute_locked(store, plan, grant, resume=journal.exists())


def finish_pending_locked(store, grant: ReclamationGrant) -> None:
    """Finish known journal transactions before accepting another publication."""
    # Only completed status parsing is cached. Every discovery stats the directory
    # entries afresh; replacement/in-place edits invalidate via full inode identity.
    # No graph, payload integrity or deletion permission is cached across writers.
    cache = getattr(store, '_completed_journal_status', {})
    current = {}
    journal_root = store.store_root / 'reclamation'
    if journal_root.exists():
        _plain(journal_root)
    for path in sorted((store.store_root / 'reclamation').glob('*.plan.json')):
        status = path.with_name(path.name.replace('.plan.json', '.status.json'))
        if status.exists():
            info = status.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Unknown completion status ownership.')
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if cache.get(str(status)) == identity:
                current[str(status)] = identity
                continue
            value = json.loads(status.read_text())
            if not isinstance(value, dict) or value.get('plan_id') != path.name.split('.')[0]:
                raise ValueError('Invalid reclamation completion status.')
            if value.get('phase') == 'complete':
                current[str(status)] = identity
                continue
        _execute_locked(store, json.loads(path.read_text()), grant, resume=True)
    # Bound process memory independently of the disk quota. Older entries simply
    # undergo fresh parsing again; never extend trust to untracked records.
    store._completed_journal_status = dict(list(current.items())[-4096:])


def rotate_locked(store, grant: ReclamationGrant) -> dict:
    """Authorized boundary operation; no reclamation happens without a grant."""
    finish_pending_locked(store, grant)
    with reader_admission(store.store_root):
        plan = _build_locked(store)
    if plan['inventory']['blocked']:
        raise RetentionLimitReached('Reclamation blocked: ' + str(plan['inventory']['uncertainties']))
    if plan['inventory']['retirement_candidates']:
        return _execute_locked(store, plan, grant)
    return dict(phase='not-needed', physical=plan['inventory']['physical'])
