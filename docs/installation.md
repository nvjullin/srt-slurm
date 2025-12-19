# Installation

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Python 3.10+
- Container runtime (enroot/pyxis) configured on the cluster
- Model weights accessible from compute nodes
- SGLang container image (`.sqsh` format)

## Clone and Install

```bash
git clone https://github.com/ishandhanani/srt-slurm.git
cd srt-slurm
pip install -e .
```

## Gather your cluster user and target partition

```bash
# user
sacctmgr -nP show assoc where user=$(whoami) format=account
# partition
sinfo
```

## Run Setup

```bash
make setup ARCH=aarch64  # or ARCH=x86_64
```

The setup will:

1. Download NATS/ETCD binaries for your architecture
2. Prompt you for cluster settings:
   - SLURM account (default: `restricted`)
   - SLURM partition (default: `batch`)
   - GPUs per node (default: `4`)
   - Time limit (default: `4:00:00`)
3. Create `srtslurm.yaml` with your settings
4. Auto-detect and set `srtctl_root` path

Dynamo 0.7.0 is now available on PyPI and will be installed automatically from pip when workers start.

## Configure srtslurm.yaml

After setup, edit `srtslurm.yaml` to add model paths, containers, and cluster-specific settings:

### Adding Model Paths

The `model_paths` section maps short aliases to full filesystem paths:

```yaml
model_paths:
  deepseek-r1: "/mnt/lustre/models/DeepSeek-R1"
  deepseek-r1-fp4: "/mnt/lustre/models/deepseek-r1-0528-fp4-v2"
  llama-70b: "/mnt/lustre/models/Llama-3-70B"
```

Models must be accessible from all compute nodes (typically on a shared filesystem like Lustre or GPFS).

### Adding Containers

The `containers` section maps version aliases to `.sqsh` container images:

```yaml
containers:
  latest: "/mnt/containers/lmsysorg+sglang+v0.5.5.sqsh"
  stable: "/mnt/containers/lmsysorg+sglang+v0.5.4.sqsh"
```

To create a container image from Docker:

```bash
enroot import docker://lmsysorg/sglang:v0.5.5
mv lmsysorg+sglang+v0.5.5.sqsh /mnt/containers/
```

### Cloud Sync (Optional)

To sync benchmark results to S3-compatible storage:

```yaml
cloud:
  endpoint_url: "https://s3.example.com"
  bucket: "benchmark-results"
  prefix: "my-team/"
```

Then use `make sync-to-cloud` or `make sync-run RUN_ID=<run_id>`.

### Cluster Compatibility Settings

Some SLURM clusters don't support certain SBATCH directives. If you encounter errors during job submission, you may need to adjust these settings:

#### GPU Resource Specification

If you see this error when submitting jobs:

```
sbatch: error: Invalid generic resource (gres) specification
```

Your cluster doesn't support the `--gpus-per-node` directive. Disable it in `srtslurm.yaml`:

```yaml
use_gpus_per_node_directive: false
```

This will omit the `#SBATCH --gpus-per-node` directive from generated job scripts while keeping all other functionality intact.

#### Segment-Based Scheduling

If you see this error when submitting jobs:

```
sbatch: error: Invalid --segment specification
```

Your cluster doesn't support the `--segment` directive for topology-aware scheduling. Disable it in `srtslurm.yaml`:

```yaml
use_segment_sbatch_directive: false
```

The `--segment` directive ensures all allocated nodes are within the same network segment/switch for optimal interconnect performance between prefill and decode workers. If your cluster doesn't support it, SLURM will still allocate nodes but may scatter them across the cluster.

## Create a Job Config

Create `configs/my-job.yaml`:

```yaml
name: "my-benchmark"

model:
  path: "deepseek-r1" # Uses alias from srtslurm.yaml
  container: "latest" # Uses alias from srtslurm.yaml
  precision: "fp8"

extra_mount: # add this if you need to mount extra directories to the container
  - "/local-dir1:/container-dir1"
  - "/local-dir2:/container-dir2"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 2
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

slurm:
  time_limit: "02:00:00"

backend:
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.84
      tensor-parallel-size: 4
    decode:
      kv-cache-dtype: "fp8_e4m3"
      mem-fraction-static: 0.83
      tensor-parallel-size: 8
      expert-parallel-size: 8
      data-parallel-size: 8
      enable-dp-attention: true

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]
  req_rate: "inf" # Request rate, use "inf" for max throughput
```

### Backend Options

The `backend` section supports these additional options:

```yaml
backend:
  # Frontend architecture (disaggregated mode only)
  enable_multiple_frontends: true # Default: true. Use nginx + multiple frontends
  num_additional_frontends: 9 # Default: 9. Additional frontends beyond the master

  # Alternative: Use SGLang router instead of nginx + frontends
  use_sglang_router: false # Default: false. Use sglang_router for load balancing
```

## Profiling (torch / nsys)

You can enable profiling via a top-level `profiling` section in your job YAML:

```yaml
profiling:
  type: "torch" # one of: "none", "torch", "nsys"
  isl: 1024
  osl: 128
  concurrency: 24
  start_step: 0 # optional
  stop_step: 50 # optional

benchmark:
  type: "manual" # Required - profiling and benchmarking are mutually exclusive
```

See [Profiling](profiling.md) for detailed configuration options, constraints, and output file locations.

## Validate with Dry Run

Always validate before submitting:

```bash
srtctl dry-run -f configs/my-job.yaml
```

This validates your config, resolves aliases, generates all files, and saves them to `dry-runs/` without submitting to SLURM.

## Submit the Job

```bash
srtctl apply -f configs/my-job.yaml
```

Output:

```
Submitted batch job 12345
Logs: logs/12345_1P_4D_20251122_143052/
```

### Submit with Tags

You can tag runs for easier filtering in the dashboard:

```bash
srtctl apply -f configs/my-job.yaml --tags experiment,baseline,v2
```

Tags are saved in the job metadata and can be used to filter runs in analysis.

See [Monitoring](monitoring.md) for how to monitor your job and understand the detailed log structure.

## Custom Setup Scripts

You can run custom initialization scripts on worker nodes before starting SGLang workers. This is useful for:

- Setting up custom environment variables
- Installing additional dependencies
- Checking out custom code

### Creating a Setup Script

1. Create your setup script in the `configs/` directory:

   ```bash
   # configs/custom-setup.sh
   # Example of checking out a specific branch of SGLang
   #!/bin/bash
    cd /sgl-workspace/
    rm -rf sglang
    git clone https://github.com/sgl-project/sglang.git
    cd sglang
    git checkout origin/cheng/refactor/sbo
    git config --global --add safe.directory "*"
    pip install -e "python"
   ```

2. Make it executable:

   ```bash
   chmod +x configs/custom-setup.sh
   ```

3. Submit with the `--setup-script` flag:
   ```bash
   srtctl apply -f configs/my-job.yaml --setup-script custom-setup.sh
   ```

The script will be executed on each worker node (prefill, decode, and aggregated) before installing Dynamo from PyPI and starting the SGLang workers. The script must be located in the `configs/` directory, which is mounted into containers at `/configs/`.

**Note**: Setup scripts only run when you explicitly specify `--setup-script`. No default setup script will run if this flag is omitted.

## Complete srtslurm.yaml Reference

Here's a complete example of all available options:

```yaml
# Default SLURM settings
default_account: "your-account"
default_partition: "batch"
default_time_limit: "4:00:00"

# Resource defaults
gpus_per_node: 4

# SLURM directive compatibility
use_gpus_per_node_directive: true # Set false if cluster doesn't support --gpus-per-node
use_segment_sbatch_directive: true # Set false if cluster doesn't support --segment

# Path to srtctl repo root (auto-set by make setup)
srtctl_root: "/path/to/srtctl"

# Model path aliases
model_paths:
  deepseek-r1: "/models/DeepSeek-R1"
  llama-70b: "/models/Llama-3-70B"

# Container aliases
containers:
  latest: "/containers/sglang-latest.sqsh"
  stable: "/containers/sglang-stable.sqsh"

# Cloud sync settings (optional)
cloud:
  endpoint_url: "https://s3.example.com"
  bucket: "benchmark-results"
  prefix: "my-team/"
```
