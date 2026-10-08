"""Stage 2 SFT data selection and visible progress helpers."""

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
from torch.utils.data import Dataset

from llm_tokenizer.sft import ChatMessage, IGNORE_LABEL, serialize_conversation


def atomic_json(path: Path, value: dict) -> None:
    """Replace one owned progress document atomically."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class ChatDataset(Dataset):
    """Seeded, source-balanced complete conversations with shifted assistant targets."""

    def __init__(self, path: Path, tokenizer, *, context: int, limit: int, seed: int) -> None:
        sources = defaultdict(list)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                sources[row['source']].append(row)
        rng = random.Random(seed)
        for rows in sources.values():
            rng.shuffle(rows)
        selected = []
        while len(selected) < limit and any(sources.values()):
            for source in sorted(sources):
                if sources[source] and len(selected) < limit:
                    selected.append(sources[source].pop())
        rng.shuffle(selected)
        self.items = []
        self.groups = {r['duplicate_group'] for r in selected}
        self.sources = {source: sum(r['source'] == source for r in selected) for source in sources}
        for row in selected:
            serialized = serialize_conversation([ChatMessage(m['role'], m['content']) for m in row['messages']],
                                                 tokenizer, maximum_tokens=context)
            ids = list(serialized.input_ids)
            labels = [*serialized.labels[1:], IGNORE_LABEL]
            pad = context - len(ids)
            self.items.append((torch.tensor(ids + [tokenizer.token_to_id('<pad>')] * pad),
                               torch.tensor(labels + [IGNORE_LABEL] * pad)))
        if not self.items:
            raise ValueError('No accepted conversational data.')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def start_chat_viewer(status: Path) -> subprocess.Popen:
    """Require automatic visible Kitty attachment before any training update."""
    if not shutil.which('kitty') or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('A visible Kitty viewer is required for training.')
    root = Path(os.environ.get("LLM_INTEGRATION_ROOT", Path.cwd())).resolve()
    process = subprocess.Popen(['kitty', '--override', 'remember_window_size=no',
                                '--override', 'initial_window_width=100c', '--override', 'initial_window_height=22c',
                                '--title', 'Codexa Stage 2 conversational SFT',
                                sys.executable, '-m', 'llm_training.conversational_training',
                                '--view', str(status.resolve())], cwd=root)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if status.with_suffix('.attached').exists():
            return process
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError('Kitty did not attach the required progress viewer.')


def view_chat(status: Path) -> None:
    """Display measured step speed, ETA, losses, tokens and checkpoint path."""
    from llm_training.viewer import render_progress
    atomic_json(status.with_suffix('.attached'), {'pid': os.getpid()})
    while True:
        value = json.loads(status.read_text())
        display = render_progress(value).replace('SPECIALIST  |  FROZEN EMBEDDING CLASSIFICATION',
                                                  'CODEXA  |  GENERAL CONVERSATIONAL SFT')
        display = display.replace('HEAD        ', 'MODEL       ').replace('FP32 head | tokens represent cached requests presented to the head',
                                                                         'BF16 decoder | speed counts supervised assistant target tokens')
        print('\033[2J\033[H' + display, flush=True)
        if value['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED', 'EARLY_STOPPED'):
            input('\nPress Enter to close.')
            return
        try:
            os.kill(value['pid'], 0)
        except ProcessLookupError:
            input('\nTraining process exited. Press Enter to close.')
            return
        time.sleep(1)


def main() -> None:
    """Display an owned Stage 2 progress document without central scripts."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    view_chat(parser.parse_args().view)


if __name__ == "__main__":
    main()
