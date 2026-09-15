# STAMP — Spatial Staleness-Aware Model-Predictive Tracking

Four fixed pan-tilt-zoom cameras have to keep eight moving targets in view on a
2000-unit square with nine obstacles (`MATE-4v8-9`). Each camera sees only what
its own sector contains — about 14% of the targets — and can talk only to the
teammates within 1500 units. The team is scored on mean coverage rate: the
fraction of targets someone is watching, averaged over the episode.

STAMP solves it without an actor and without a critic. The policy *is* a
planner, recomputed from scratch at every decision, and the only thing with
weights is a generative model of where the targets are going.

## One decision, end to end

1. **Peer-to-peer round.** Every camera addresses its target-slot block to each
   neighbour in range — not a broadcast. Slot *j* is the same target for every
   camera, so merging is a per-slot choice and needs no data association.
2. **Belief.** The merge plus two columns no observation can provide: how stale
   each slot is, and whether this camera saw it first hand. 56 dimensions here
   (8 targets × 7 fields). Without the first-hand bit a fully connected team
   holds identical beliefs and the consensus loss goes to exactly zero.
3. **Trajectory flow.** A conditional flow-matching head samples where every
   believed target goes over the next four decisions — one sample is one
   coherent future for all eight, not a per-target average.
4. **MPPI.** 64 candidate command sequences are rolled through MATE's exact
   camera optics (the sector area `θ·Rs²` is conserved, so range and width
   trade off in closed form), scored by soft coverage, and the softmax-weighted
   average is executed.
5. **Intent.** The executed plan publishes what it means to cover, on the same
   channel that carries the belief, so neighbours discount those targets on
   their next decision. That is the whole coordination mechanism.

## What is learned, and what is not

| Component | Models | Learned |
|---|---|---|
| Camera optics in the rollout | what a command does to the sector | no — MATE's own closed form |
| Flow trajectory head | where the targets go next | **yes** — the only weights in the loop |
| Staleness map | where the team has not looked | no — counted |

The learning signal is dense and supervised: true future target positions, read
out of the global state at training time only, regressed through a flow-matching
velocity field. No scalar team reward is ever backpropagated.

## Active search — what this repository adds

The tracking reward weights a target by whether a camera believes in it, which
means every target nobody has found yet is worth exactly zero, and nothing pays
for going to look. Measured on this scenario, the whole team only ever knows
about half the targets between them, so that hole is most of the remaining
coverage.

Three terms were added to the planner's reward to close it. All three are
reward, not weights — a configuration can be scored on a checkpoint trained
under a different one, and setting every weight to zero reproduces the previous
policy exactly.

- **`explore_weight`** — a 16×16 staleness map, where each cell is worth how
  many decisions it has gone unseen (saturating at 40), scored by the same soft
  coverage the targets get. Each camera zeroes the cells its own sector covers
  *and* those its in-range neighbours covered, so the map is per camera and
  honest about the communication range. Cells are split between cameras by
  Voronoi: without the split every camera reads the same map and they all turn
  towards the same stale corner. **This is the term that pays.**
- **`recall_weight`** — a lost target is somewhere on a ring of radius
  `drift × age` around its last fix (17 units per decision, measured). The ring
  is sampled at four points and widens over the roll. Measured at +0.043, inside
  the noise of the protocol; ships at zero.
- **`angle_weight`** — opening the aperture when there is little left to track.
  Same story: real on one seed, gone on the others; ships at zero.

## Results

`MATE-4v8-9`, mean coverage rate.

| Policy | Coverage |
|---|---|
| random rule | 0.2854 |
| MAPPO, model-free, 150k steps | 0.3405 |
| naive rule | 0.3669 |
| MPPI planning, untrained head | 0.4739 |
| MPPI planning, tracking reward only | 0.5056 |
| heuristic rule | 0.5934 |
| greedy rule | 0.6028 |
| greedy rule + oracle field of view | 0.6372 |
| **MPPI planning + active search** | **0.6701** |

Two results worth stating plainly, because they are the ones a reviewer will
ask about:

- Active search beats an oracle that hands the planner the true positions of all
  eight targets (+0.110). An oracle still makes a camera point at points; the
  staleness term makes it cover area, and area is what produces the next
  detection.
- The flow head, trained and converging, scores the same as assuming every
  target stands still. The generative model is honest machinery that this
  scenario does not reward — the correct baseline for a sampling head is
  persistence, not an untrained head, and against that baseline it does not win.

## Running it

Five steps, in order. Every command below was run to produce the numbers above.

**1. Install.** Python 3.13, and `MATE-main` sitting next to this README (it is a
source checkout, not a package — nothing installs it):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Anaconda, `import torch` after numpy can abort with `OMP: Error #15`. Export
`KMP_DUPLICATE_LIB_OK=TRUE`, or use a clean venv rather than the base
environment.

**2. Check the install** — one planned decision and one training step, asserted,
about a minute:

```bash
python check_pipeline.py --device cpu
```

It must print `belief (4, 56)`, `trajectory (4, 8, 4, 2)`,
`search cells (256, 2) | staleness (4, 256)` and end with `OK on cpu`. A
different belief width or communication range means the configuration is not the
one this README describes.

**3. Reproduce the comparison table** — every method, one protocol, same episode
seeds. Twenty episodes takes about twenty minutes on a laptop CPU:

```bash
python baseline.py --episodes 20
```

Add `--checkpoint runs/<tag>/seed<N>/best.pt` and the table gains two more rows:
the trained planner, and the same weights with the search terms switched off.
That second row is the ablation, and it costs nothing extra to produce, because
the search terms live in the planner's reward rather than in any weights.

Add `--matrix` instead to re-run each rule agent with and without MATE's message
round and with and without oracle field-of-view fusion, with paired deltas — the
measurement that says what information sharing is worth.

**4. Train.** 50k environment steps is the working budget; the flow head
converges well before it and nothing else in the loop learns:

```bash
python train.py --seed 1 --tag v1 --steps 50000 --device cuda --final-episodes 50
```

A run writes `config.yaml`, `metrics.jsonl`, `checkpoint.pt`, `best.pt` and
`results.json` under `runs/v1/seed1/`. `results.json` is the number to report.
There is no resume — runs are short, restart from zero.

**5. Score a configuration without retraining:**

```bash
python search_eval.py --checkpoint runs/v1/seed1/best.pt --episodes 20 \
    --set planner.explore_weight=0.1
```

Any `planner.*` key can be overridden this way, on a checkpoint trained under a
different one.

### Before you believe a difference

The flow head samples, so scoring the same checkpoint, the same configuration and
the same episode seeds twice does not give the same number. Measured over four
repeats at ten episodes: 0.6269, 0.6493, 0.6569, 0.6743 — a spread of 0.047.
**Anything under about 0.05 at ten episodes is noise.** Use fifty episodes and
several seeds before believing a difference, and watch `acquisition` and `union`
rather than coverage: they move earlier and spread less.

Operational details — the full measurement protocol, what every logged metric
should look like, GPU notes, wandb, and the known traps in MATE's seeding — are
in [RUNNING.md](RUNNING.md).

## Layout

```
train.py              training loop; --set overrides any config leaf
evaluate.py           deterministic-seed evaluation of a checkpoint
baseline.py           every method under one protocol, one table
search_eval.py        score a search configuration on an existing checkpoint
check_pipeline.py     one decision and one training step, asserted

src/envs/             MATE wrapper, P2P channel, belief, spatial memory
src/models/           flow-matching trajectory head and its two losses
src/algorithms/       MPPI planner, rollout collector, buffer
src/configs/          default.yaml — every knob, with why it is set that way
```

Installation, the verification sequence, the measurement protocol and the known
traps are in [RUNNING.md](RUNNING.md). The simulator itself is not part of this
repository; `MATE-main` is a checkout of it.
