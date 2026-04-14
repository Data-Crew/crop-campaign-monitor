# Runtime Stack

> Docker Compose runtime model, GPU passthrough, and environment variable
> reference for the Crop Campaign Monitor.

---

## Runtime Model

The environment is split into two types of services defined in
`docker-compose.yml`:

| Type | Services | Behaviour |
|------|----------|-----------|
| **Persistent** | `workspace`, `dashboard` | Started by `docker compose up -d`. Remain running until stopped. |
| **One-shot jobs** | `data-prep`, `training`, `monitor`, `pipeline` | Run with `docker compose run --rm <service>`. Exit when the phase completes. Require `--profile jobs` only for `docker compose up`; `run` works without it. |

All services share the same image, bind mounts, and GPU wiring. No container
names or IDs need to be discovered manually — every interaction uses a stable
service name.

---

## Starting the Environment

Start both persistent services (workspace + dashboard):

```bash
docker compose up -d
```

Start only a specific persistent service:

```bash
docker compose up -d dashboard    # dashboard only
docker compose up -d workspace    # workspace shell only
```

Stop everything:

```bash
docker compose down
```

---

## Opening a Shell

The `workspace` service runs a persistent bash process. Attach a terminal to
it at any time:

```bash
docker compose exec workspace bash
```

From inside the shell you can run scripts, inspect data, edit configs, or
invoke individual pipeline steps directly.

---

## Running the Dashboard

The `dashboard` service starts Streamlit automatically when the container
starts. No manual launch step is required.

```bash
docker compose up -d dashboard
```

Open `http://localhost:8501` in your browser. The sidebar guides you through
the three phases with status indicators and run buttons.

To view dashboard logs:

```bash
docker compose logs -f dashboard
```

---

## Running Pipeline Phases

Each phase job runs in a fresh container derived from the same image, with
the same GPU wiring and bind mounts as the persistent services. The container
is removed automatically after the phase completes (`--rm`).

```bash
# Phase 1 — Data Preparation (ingest → fetch → chip)
# Must run at least once before Phase 2 or Phase 3.
docker compose run --rm data-prep

# Phase 2 — Training (prepare → finetune → export_encoder)  [optional]
# Requires Phase 1 outputs (parcels_labeled.parquet + data/chips/).
docker compose run --rm training

# Phase 3 — Monitor (embed → profile → score → report → index → explain)
# Requires Phase 1 outputs (data/chips/).
docker compose run --rm monitor

# Full pipeline — all three phases in sequence
docker compose run --rm pipeline
```

### Passing a custom config or overrides

Arguments after the service name are forwarded to the underlying script:

```bash
# Use a non-default monitor config
docker compose run --rm data-prep config/my_region.yaml

# Pass config overrides (key=value syntax)
docker compose run --rm monitor config/monitor.yaml season.start_date=2024-05-01
```

### Running individual steps

Use `docker compose exec workspace` to run a single step inside the
persistent workspace container:

```bash
# Phase 1
docker compose exec workspace bash scripts/run_step.sh ingest
docker compose exec workspace bash scripts/run_step.sh fetch
docker compose exec workspace bash scripts/run_step.sh chip

# Phase 2
docker compose exec workspace bash scripts/run_step.sh prepare  config/train.yaml
docker compose exec workspace bash scripts/run_step.sh finetune config/train.yaml
docker compose exec workspace bash scripts/run_step.sh export   config/train.yaml

# Phase 3
docker compose exec workspace bash scripts/run_step.sh embed
docker compose exec workspace bash scripts/run_step.sh profile
docker compose exec workspace bash scripts/run_step.sh score
docker compose exec workspace bash scripts/run_step.sh report
docker compose exec workspace bash scripts/run_step.sh index
docker compose exec workspace bash scripts/run_step.sh explain
```

---

## GPU Access

### Default approach: manual device and library mounts

The default GPU passthrough strategy mounts NVIDIA device nodes and host
libraries directly into the container. This works without any additional
host-side tooling beyond standard NVIDIA drivers.

Relevant section of `docker-compose.yml` (via the shared `x-gpu-config` anchor):

```yaml
volumes:
  - /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4:/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4:ro

devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidia1:/dev/nvidia1    # remove if host has only one GPU
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
  - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
```

Adjust the `devices` list to match your host GPU topology. To add a third
GPU, add `/dev/nvidia2:/dev/nvidia2`. To use only one GPU, remove
`/dev/nvidia1`.

### GPU selection by name

GPUs are selected **by name substring** (not by device index) so the correct
GPU is always used regardless of device ordering after a reboot.

Configure in `config/monitor.yaml` (Phases 1 and 3):

```yaml
gpu:
  default_device: "RTX 4070"
  smoke_test_device: "RTX 500"
```

Configure in `config/train.yaml` (Phase 2):

```yaml
gpu:
  device: "RTX 4070"
```

Override at runtime without editing the config files:

```bash
GPU_NAME="RTX 4070" docker compose run --rm data-prep
```

The Streamlit dashboard discovers all available GPUs at startup and displays
them by name and VRAM so the selection can be confirmed before running a phase.

---

## GPU Topology Changes

GPU device nodes (`/dev/nvidia*`) are mounted into containers at startup. If
the host GPU topology changes after the containers are already running — for
example, an eGPU is reconnected, a driver is updated, or a new GPU is added —
the device nodes inside the running containers will be stale.

**Always recreate the containers after a host GPU topology change:**

```bash
docker compose down
docker compose up -d
```

Then re-launch any job services needed. Running a pipeline phase against a
container with stale GPU mounts will typically produce CUDA errors.

---

## Environment Variables

Set these in a `.env` file alongside `docker-compose.yml`, or export them in
your shell before running `docker compose up`.

| Variable | Default in Compose | Purpose |
|----------|--------------------|---------|
| `GPU_NAME` | `RTX 4070` | Selects the GPU by name substring match. Overrides `gpu.device` / `gpu.default_device` from the config files. |
| `MAPBOX_API_ACCESS_TOKEN` | _(public demo token)_ | Mapbox basemap token for the Streamlit dashboard. Replace with your own token for production use. |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Makes all GPU device nodes visible inside the container. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Enables CUDA compute and `nvidia-smi` inside the container. |
| `LD_LIBRARY_PATH` | `/usr/lib/x86_64-linux-gnu` | Points the dynamic linker to the host-mounted NVIDIA libraries. |

Example `.env` file:

```env
GPU_NAME=RTX 4070
MAPBOX_API_ACCESS_TOKEN=pk.your_own_token_here
```

---

## Optional Advanced Alternative: NVIDIA Container Toolkit

> This section describes an optional alternative runtime approach.
> It is not the default supported configuration for this repository
> and is not required for normal usage.

If `nvidia-container-toolkit` is installed and configured on the host, it can
manage GPU passthrough automatically through a Docker runtime plugin. This
eliminates the need to enumerate device nodes and library paths manually.

To use it, remove the `devices` list and the four NVIDIA library entries from
`volumes` in `x-gpu-config`, and add a `deploy` block to each service that
requires GPU access:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

This approach requires the host to have `nvidia-container-toolkit` correctly
installed and the Docker daemon configured to use the NVIDIA runtime. Consult
the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
for host-side setup.

The manual mount strategy documented above is the baseline that is known to
work with this repository. If you switch to the toolkit approach, validate
GPU access with `nvidia-smi` inside the container before running any pipeline
phase.
