# Configs

Configuration split:

- `datasets/`: dataset stage maps, completeness level, and multi-key guidance
- `models/`: backbone and reflection-model settings
- `runs/`: experiment-level overrides for main tables and ablations

The goal is to replace scattered per-script constants with explicit config files
and to make `MARCO`, `NQ`, and `TQA` differences explicit instead of implicit.
