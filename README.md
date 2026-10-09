# LLM-Training

Base and conversational training, recoverable checkpoints and monitoring.

Owns optimizers, precision-aware loops, logging, combined YAML config, resume/checkpoint writing and monitoring. Full/native/legacy SFT artifacts retain separate support. Legacy SFT resume does not restore optimizer/RNG state. Stage 2 and overfit orchestration remains in the integration repository because it evaluates/exports across components. Tiny automated tests are not production training launches.

## Development

In the sibling workspace, use `../LLM-From-Scratch/run.py --repo LLM-Training test`.
This selects the existing environment and sibling package sources without installing dependencies.
For a separately installed checkout, run `python -m pytest` after provisioning the documented dependencies and exact sibling version 0.1.0. These packages are local and not published to PyPI.

## Entry points

- `python -m llm_training.cli.train --help`
- `python -m llm_training.cli.train_conversational_sft --help`
- `python -m llm_training.cli.monitor_gpu --help`
- `python -m llm_training.cli.summarize_gpu --help`
- `python -m llm_training.cli.summarize_training --help`

## Integration and assets

`../LLM-From-Scratch/compatibility.json` records the complete tested version set.
Checkpoint weights, tokenizers, datasets and generated logs are referenced by path; none are distributed in this package. Preserve tokenizer fingerprints and architecture lineage. Source provenance is in PROVENANCE.md.

## Validation and limitations

See the central VALIDATION.md for commands, results and unverified large-model checks.
The original project is preserved unchanged. No model promotion, training pipeline or remote publishing occurs as part of extraction.

# LLM-Training-PyTorch-Codexa-v1

## Canonical workspace integration

This repository remains independently versioned at its existing remote and is pinned as a sibling in LLM-From-Scratch/compatibility.json. Integration decisions live in ../LLM-From-Scratch/documentation/training/SESSION_DECISIONS.md. Historical assets are external inputs; never commit weights, datasets or recovery snapshots.
