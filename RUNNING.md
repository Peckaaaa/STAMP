# Running this repository on a server

Operational guide: how to install, how to check the install is sane, what to run
to produce comparable numbers, and what the known traps are. The architecture is
documented in the module docstrings — read them in pipeline order:
`src/envs/mate_wrapper.py` → `src/models/trajectory.py` →
`src/algorithms/mppi_planner.py` → `train.py`.

## The pipeline in one paragraph

Each camera sends its observed target slots to the neighbours inside
`comm_range` and merges what comes back into a **belief** (56 dims for
`MATE-4v8-9`: eight targets × `[x, y, sight range, loaded, first hand, age,
known]`). A **conditional flow-matching head** samples where every believed
target goes over the next four decisions. An **MPPI planner** rolls 64 candidate
command sequences through MATE's exact camera optics, scores each by soft
coverage of that sample, and executes the softmax-weighted average; the executed
plan publishes what it intends to cover so neighbours discount it next step.
There is no actor and no critic — the policy is the planner, recomputed every
decision, and the flow head is the only thing with weights.

---

## 1. What the repository needs

The simulator is **not** in this repository. `src/envs/config_resolver.py`
imports `mate` from a MATE-main checkout, searching in this order:

1. `$MATE_ROOT`
2. `./MATE-main`
3. `../MATE-main`
4. `../MATE/MATE-main`

MATE-main is a **gymnasium** port. The old `gym` (0.26) package is not usable
with it. `pip install -e MATE-main` is not required — putting it on the path is
enough, because `mate` only needs gymnasium, numpy, scipy and pyyaml at import
time.

Verified combination: Python 3.13, gymnasium 1.2.3, numpy 2.4, scipy 1.11,
torch 2.12.

---

## 2. Install

```bash
git clone <this repository> mate-diffusion
cd mate-diffusion

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install wandb                    # only if you want the dashboard

export MATE_ROOT=/abs/path/to/MATE-main    # if it does not sit inside the repo
```

On a GPU box install the CUDA build of torch rather than the default wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`cuda False` on a GPU machine means the CPU wheel got installed; the run will
fall back silently, which is why `train.py` prints the resolved device.

---

## 3. Check the install (about one minute)

```bash
python check_pipeline.py --device cpu
python check_pipeline.py --device cuda      # on the GPU box
```

This runs one planned decision and one training step and asserts both the device
placement and the tensor layout, including the one that is easy to get wrong:
the buffer stores target positions time-major and the head predicts
target-major, so a training window must come back `(B, 8, 4, 2)`. Expected
output on `MATE-4v8-9`:

```
MATE-4v8-9.yaml: 4 cameras, 8 targets, 9 obstacles | ... P2P payload 48, belief 56 ... comm range 1500
  belief      (4, 56)   cpu
  trajectory  (4, 8, 4, 2)  cpu
  believed    (4, 8)    cpu
  actions     (4, 2)    cpu
  intent      (4, 8)    cpu
  window      beliefs (32, 4, 56) | labels (32, 8, 4, 2)
  update      prediction_loss=... consensus_loss=... rmse_units=... believed_fraction=...
OK on cpu
```

If the header says a different belief dimension or communication range, the
configuration is not the one these instructions describe.

---

## 4. Smoke test (about one minute)

```bash
# CPU
python train.py --seed 0 --steps 1000 --final-episodes 2 --no-wandb --device cpu --tag smoke \
  --set train.env_steps_per_iter=500 --set train.warmup_steps=200 \
  --set train.trajectory_updates_per_iter=2 --set train.eval_every=1 --set train.eval_episodes=2

# GPU, same run, plus wandb so the logging path is exercised too
python train.py --seed 0 --steps 1000 --final-episodes 2 --wandb --device cuda --tag smoke \
  --set train.env_steps_per_iter=500 --set train.warmup_steps=200 \
  --set train.trajectory_updates_per_iter=2 --set train.eval_every=1 --set train.eval_episodes=2
```

Both must print two iteration lines and a `final:` line, and leave five files in
`runs/smoke/seed0/`. Check the log file parses before trusting a long run:

```bash
python -c "
import json
rows = [json.loads(l) for l in open('runs/smoke/seed0/metrics.jsonl')]
print(len(rows), 'rows'); print(sorted(rows[-1]))"
```

Then delete `runs/smoke`.

---

## 5. Producing the numbers

Episode spread is large (0.08–0.15 coverage), so at 50 evaluation episodes the
standard error is roughly 0.02 and a difference under about 0.05 needs several
seeds before it means anything. Run five.

```bash
for seed in 1 2 3 4 5; do
  python train.py --seed $seed --tag v1 --steps 50000 --device cuda --final-episodes 50
done
```

50k steps is the working budget: the flow head converges long before it, and
nothing else in the loop learns. Run 100k if you want the curve flat for a
figure rather than merely converged.

### Rule-based reference points

These need no training and set the floor and the ceiling of the whole study:

```bash
python phase0.py --episodes 30 --out runs/v1/phase0.json
```

It reports `random`, `naive`, `greedy` and `heuristic` under four settings each
(with and without MATE's own message round, with and without oracle
field-of-view fusion) plus the paired deltas — roughly 40 minutes for 30
episodes. Every configuration replays the same episode seeds, so the deltas are
paired rather than two independent samples.

Score a trained run under exactly the same protocol:

```bash
python phase0.py --episodes 30 --agents greedy --checkpoint runs/v1/seed1/best.pt
python evaluate.py --checkpoint runs/v1/seed1/best.pt --episodes 50
```

### Reference numbers already measured

`MATE-4v8-9`, 20-episode protocol, single seed each. Anything new should be
compared against these:

| Policy | Coverage |
|---|---|
| random rule | 0.2854 |
| MAPPO, model-free, 150k steps | 0.3405 |
| naive rule | 0.3669 |
| MPPI planning, untrained head, 10k steps | 0.4739 |
| MPPI planning, MLP head, 30k steps | 0.5474 |
| heuristic rule | 0.5934 |
| greedy rule | 0.6028 |
| greedy rule + oracle field of view | 0.6372 |

---

## 6. Wall-clock budget

Measured on a 4-thread CPU box, `MATE-4v8-9`: 55–95 environment steps per
second, so 50k steps is 10–15 minutes.

Where the time goes, and what a GPU changes:

- The simulator is numpy and single-threaded. It is the floor on throughput and
  a GPU does not touch it.
- Everything downstream of the belief — sampling the flow, rolling 64
  candidates, scoring coverage, the training step — is batched torch and moves
  to the GPU with `--device cuda`. On this problem size that is a small part of
  the per-step cost, so **expect a modest speedup, not a large one**. The GPU
  matters when you scale `planner.samples`, `trajectory.horizon`, the number of
  cameras, or `trajectory.hidden_dim`.
- Because the bottleneck is the simulator, the efficient way to use a big
  machine is many seeds in parallel, one thread each:

```bash
mkdir -p logs
for seed in 1 2 3 4 5; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup \
    python -u train.py --seed $seed --tag v1 --steps 50000 --device cuda --final-episodes 50 \
    > logs/seed$seed.log 2>&1 &
done
wait
```

Use `tmux` or `nohup` — there is **no resume**. Checkpoints are written as the
run goes, but the buffer is not serialized, so restarting from one would
silently be a different experiment. Runs are short; restart from zero.

---

## 7. What a run writes

```
runs/<tag>/seed<N>/
├── config.yaml      the fully resolved configuration, including the seed
├── metrics.jsonl    one JSON object per iteration
├── checkpoint.pt    latest weights
├── best.pt          weights at the best periodic evaluation
└── results.json     the final evaluation over --final-episodes episodes
```

`results.json` is the number to report. Aggregating a grid:

```bash
python - <<'PY'
import json, glob, statistics, collections
rows = collections.defaultdict(list)
for path in glob.glob('runs/*/seed*/results.json'):
    r = json.load(open(path))
    rows[r['tag']].append(r['coverage_rate'])
for tag, values in sorted(rows.items()):
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f'{tag:12s} {statistics.mean(values):.4f} +- {spread:.4f} over {len(values)} seeds')
PY
```

Report the across-seed spread, not the within-run episode spread: they answer
different questions and the episode one is always larger.

### The metrics, and what healthy looks like

| Key | Meaning | Healthy |
|---|---|---|
| `env/coverage_rate`, `env/coverage_rate_std` | training rollouts | rises during the run |
| `eval/coverage_rate` | periodic deterministic-seed evaluation — **the one to report** | above 0.45 within a few thousand steps |
| `traj/prediction_loss` | flow-matching velocity regression (`L_pred`) | falls; its scale is not comparable to a plain MSE head |
| `traj/consensus_loss` | disagreement between neighbouring cameras (`L_cons`, weight 0.1) | small **and** falling while `rmse_units` also falls. Agreement improving while accuracy degrades is the trivial optimum — that is the failure to catch |
| `traj/rmse_units` | sampled position error in map units, on a 2000-wide map | 25 or below over four steps |
| `traj/consensus_pairs` | camera pairs the consensus term averaged over, per batch element | above zero. Exactly zero means the term is inert — see the `comm_range` trap below |
| `belief/believed_fraction` | fraction of target slots a camera knows about | 0.25–0.45. A camera sees about 0.14 by itself, so lower means the channel is not delivering |
| `plan/max_overlap` | how much two plans claim the same target | near 1.0. Rising towards the camera count means `intent_discount` is too weak |
| `system/sps` | environment steps per second | flat; a decline means something is leaking |

---

## 8. Known traps

**Duplicate OpenMP runtime.** On Anaconda installs, importing torch after
numpy/scipy can abort with `OMP: Error #15 ... libiomp5md.dll already
initialized`. Export `KMP_DUPLICATE_LIB_OK=TRUE`, or better, use a clean venv
rather than the base conda environment.

**`comm_range` changes the meaning of the experiment.** With it unset, every
camera merges to the same union of the team's field of view: the peer-to-peer
belief collapses into a broadcast and the consensus term goes to exactly zero
(measured, to eight decimals). Cameras in `MATE-4v8-9` sit 1015 apart at the
closest and 1406 at the median, so 1500 leaves roughly two thirds of the pairs
connected. Change it only deliberately, and re-check `traj/consensus_pairs`
afterwards.

**MATE-main seeding is partly broken.** `env.seed()` and `AgentBase.seed()` both
hand a numpy integer to a gymnasium API that rejects it, and the built-in target
agents are never seeded by the environment at all. `phase0.py` works around this
by setting the agents' generators directly — do not "fix" that code by calling
the official API, it raises.

**`SharedFieldOfView` is silently dropped.** `SingleTeamHelper.__init__` walks
down to the base `MultiAgentTracking` and wraps that, so
`MultiCamera(SharedFieldOfView(base))` loses the wrapper without an error.
Oracle fusion is implemented directly in `src/envs/observation_fusion.py`.

**`displacement_scale` is tied to the scenario.** Targets move about 0.013
belief units per step per axis in `MATE-4v8-9`, and the labels are scaled by 75
so the unit-Gaussian prior is not fighting the data. A scenario with different
speeds needs this re-measured, or the flow head will spend its capacity on the
scale instead of the shape.

**Evaluation is stochastic by construction.** The head samples, so two
evaluations of the same weights differ. That is the point — a mean prediction
would send a camera down the middle of a fork — but it means small differences
need episodes, not repeats.

**Superseded modules live in `DIMA_base/`.** The actor-critic MAPPO, the
state-space world model (FSQ tokenizer, categorical diffusion, reward head) and
`diagnose.py` were moved there when the unified pipeline replaced them. Nothing
under `src/` imports them. They no longer run as they stand -- the environment
wrapper returns beliefs where they expect messages -- so treat them as a
reference copy; `DIMA_base/README.md` records what each produced and what
`diagnose.py` established.

---

## 9. wandb

`--wandb` enables it, `--no-wandb` disables it, and `train.wandb` in the config
decides when neither flag is given. Project `mate-diffusion-planning`, grouped
by `--tag`, run named `seed<N>`, and `results.json` is copied into the run
summary at the end. `metrics.jsonl` is written either way, so nothing is lost on
a cluster without outbound network access.

Offline clusters can still use the dashboard afterwards:

```bash
WANDB_MODE=offline python train.py --seed 1 --tag v1 --steps 50000 --device cuda
wandb sync runs/v1/seed1/wandb/offline-run-*      # from a machine with network
```
