# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Setup

Please run the following command to setup the srtctl tool. This repo currently requires Dynamo 0.7.0 or later. You will have to build this yourself for now since official wheels are not yet available on PyPi.

```bash
# One-time setup
make setup ARCH=aarch64  # or ARCH=x86_64
```

## Documentation

**Full documentation:** https://srtctl.gitbook.io/srtctl-docs/
