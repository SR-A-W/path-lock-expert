# LLaMA-Factory (submodule placeholder)

This directory originally contains a fork of **LLaMA-Factory** used as the training framework for PLE.

To preserve anonymity during double-blind review, the submodule content has been **removed from this repository**.

## How to reproduce

Clone the official upstream and install:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git .
pip install -e .
```

Our training configs under `../tasks/*.yaml` are designed to work with standard LLaMA-Factory (no custom modifications required for the main experiments).

## Running training

From the project root:

```bash
cd train/LLaMA-Factory
llamafactory-cli train ../tasks/<config>.yaml
```

See `../tasks/` for the full list of training configurations, and `../../slurm_examples/` for representative SLURM scripts.
