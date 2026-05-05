# TD Deployment Search Planner

This planner models the TD bundle/template search in weak-network mode. Inter-node transfer is assumed non-bottleneck, so the objective is to maximize the minimum stage throughput.

## Profile Registry

Default runs use the package-native registry in `src/aiconfigurator/td_deployment_search/profile_registry.py`.

The main dataclasses are:

- `MemoryProfile`: component memory dictionaries keyed by parallelism.
- `PEHardwareProfile`: PE TTFT/TPOT/memory for one hardware type.
- `PEModelProfile`: PE model registry plus KV-cache-per-token data.
- `StageHardwareProfile`: latency/memory for Encoder, DiT, or VAE on one hardware type.
- `GeneratorModelProfile`: generator model registry with Encoder_CPU, Encoder GPU, DiT, and VAE profiles.
- `TDProfileData`: top-level PE and generator profile map.

To add a new generator, create a `GeneratorModelProfile`, add A800/A100/H100 stage profiles for Encoder, DiT, and VAE, set `main_dit_bundle_size`, and insert it into `TDProfileData.generator_models`. A100 currently reuses A800 generator latency/memory, while H100 uses H200 NVLink latency and A800 memory, matching the profiling assumptions.

## Pruned Matrix ILP

The pruned search considers:

- `PE_Only`, only bundle size 2.
- `DiT_Only` and `DiT_VAE_Comb`, with DiT bundle size 8 for WAN2.2, 4 for WAN2.1, and 2 for Z-Image.
- `VAE_Only`, bundle size 1, 2, 4, or 8.
- `Encoder_CPU`, derived after solving.

The matrix ILP uses these variables:

- `n[h,p]`: number of nodes of hardware `h` using bundle partition `p`.
- `x[h,p,size,t]`: number of bundles assigned template `t`.
- `lambda`: bottleneck throughput.

Constraints:

- `sum_p n[h,p] <= available_nodes[h]`.
- `sum_t x[h,p,size,t] <= multiplicity(p,size) * n[h,p]`.
- `lambda <= sum x[h,p,size,t] * capacity[h,size,t,stage]` for each required stage.
- Template memory legality is prefiltered by `template_memory <= bundle_total_hbm`.

The old enumeration solver is kept as a reference backend. The runner compares the matrix and enumeration top bottleneck throughput on the pruned search before producing reports.

## Active Fragment Completion

The ILP objective is max-min throughput. It may leave a fragment inside an already-active node when that fragment cannot improve the bottleneck. For example, WAN2.2 constrains DiT to P8, so a node with `3x PE_Only_A800_P2` has a remaining P2 fragment that cannot host DiT, but can still host PE or VAE.

Reports therefore keep the base ILP core and add two fragment-filled views:

- `node-local same-model first`: fill an active-node fragment with the same template already present on that node when legal.
- `minimum wasted throughput first`: enumerate legal exact fillers for the fragment and choose the one that maximizes final bottleneck throughput; ties choose the smallest added non-bottleneck capacity.

Fragment completion never activates a completely idle node. It only consumes GPU slots inside nodes already used by the base ILP solution.

## Full Search

Full search has two variants:

- `Encoder_CPU`: all 7 non-empty templates over `{PE, DiT, VAE}`; Encoder_CPU is derived after solving.
- `Encoder_GPU`: all 15 non-empty templates over `{PE, Encoder, DiT, VAE}`.

For compound templates, memory is the sum of member stage memory and stage capacity is the per-stage profiled throughput on the same hardware/bundle size.

Before ILP, templates are Pareto-pruned per hardware and bundle size. A template is removed when another template has no lower capacity for every required stage and no higher memory, with at least one strict improvement.

## Flip Semantics

The main report prints two flip plans:

- Restricted flip plan: only `DiT_Only`, `VAE_Only`, or `DiT_VAE_Comb` can become `PE_Only`. The planner enumerates whole-bundle flips and chooses the config with maximum `min(PE, DiT, VAE)` under output tokens 2048. Ties prefer smaller `abs(PE - DiT)`, fewer flips, then less PE overproduction.
- Standard source-to-target plan: a minimum counter difference between the fragment-filled output-512 source and fragment-filled output-2048 target, including reverse moves such as `PE_Only -> DiT_VAE_Comb`.

Each detailed config also prints throughput arithmetic. Per-instance throughput is always `1 / latency`, and stage throughput is the sum of all instance throughputs for that stage. Overall throughput is the minimum required-stage throughput.

## LCM Semantics

For each PE scenario, phase 1 is output tokens 512.

Case A supplies VAE by converting the minimum number of H100 `DiT_Only_SP8` instances into `DiT_VAE_Comb_SP8`. Remaining DiT instances stay `DiT_Only`.

Case B requires VAE to be `VAE_Only`. The planner chooses the minimum positive integer node count across A800/H100 and bundle sizes 1, 2, 4, and 8. Ties prefer bundle size 8, then A800.

Phase 2 keeps the phase-1 config, changes PE output tokens to 2048, then flips H100 `DiT_Only` first and `DiT_VAE_Comb` second into H100 `PE_Only`. It chooses the discrete flip count that maximizes `min(PE, Encoder, DiT, VAE)`, then minimizes `abs(PE - DiT)`, then uses fewer flips.

## Runner

Run every report for all three generators:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_deployment_search.py
```

Run one generator:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\woshi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' tools\td_deployment_search.py --generator-model z-image
```

Useful options:

- `--generator-model`: repeatable; defaults to WAN2.2, WAN2.1, and Z-Image.
- `--pe-model`: defaults to `pe7b`.
- `--input-tokens`: defaults to `128`.
- `--top-k`: defaults to `5`.
- `--mip-time-limit-s`: solver time budget per ILP.
- `--output-dir`: report directory; defaults to `C:\aiconfigurator\td_outputs`.
- `--legacy-profile-data`: optional old-style local `data.py` adapter for development comparison.
- `--skip-backend-compare`: skip the pruned matrix/enumeration smoke comparison.
