"""Ablation study: remove one STAMP component at a time, same protocol as the full pipeline.

Two kinds of ablation, because STAMP has two kinds of component:

  eval-time   the component lives in the planner's reward or in how the
              prediction is made, not in any weights -- so it is removed on the
              full pipeline's own checkpoints, and every row is *paired* with the
              full row: same weights, same episode seeds, same torch seed per
              episode.

  train-time  the component changes what the flow head sees or how it is
              trained, so the head has to be retrained without it (train.py,
              same budget and seeds as the full pipeline), then scored here
              under the same protocol.

The full pipeline itself (`ref`) is retrained under the same budget as the
train-time variants, so both groups are compared against a budget-matched row.

The peer-to-peer channel is not ablated: every row, the full pipeline included,
runs on the same base environment -- MATE's RestrictedCommunicationRange around
MultiCamera, at the checkpoint's `env.comm_range`.

    python ablation.py list
    python ablation.py eval  --variant no_intent --checkpoint runs/stamp/seed1/checkpoint.pt \
                             --out runs/ablation/eval/no_intent/seed1.json
    python ablation.py report --root runs/ablation/eval

`scripts/run_ablation.sh` drives the whole grid.
"""

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
from gymnasium.utils import seeding
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from evaluate import EpisodeTally   # noqa: E402
from train import build_env, build_planner, build_trajectory, env_spec   # noqa: E402


# Every variant removes exactly one thing from the full pipeline.
#   kind      eval: scored on the full checkpoints; train: needs its own runs
#   set       config overrides (section.key -> value) applied before scoring,
#             or passed to train.py as --set for a train-time variant
#   predictor flow (the trained head) or persist (every target stands still)
VARIANTS = {
    'full': {
        'kind': 'ref', 'set': {}, 'predictor': 'flow',
        'label': 'STAMP (full pipeline)',
    },
    # --- stage 3: trajectory flow ------------------------------------------------
    'no_flow': {
        'kind': 'eval', 'set': {}, 'predictor': 'persist',
        'label': 'w/o flow head (persistence: targets stand still)',
    },
    # --- stage 4: MPPI scoring ---------------------------------------------------
    'no_soft': {
        'kind': 'eval',
        'set': {'planner.range_softness': 1.0e-3, 'planner.angle_softness': 1.0e-3},
        'predictor': 'flow',
        'label': 'w/o soft detection scoring (hard in/out count)',
    },
    # --- stage 5: intent ---------------------------------------------------------
    'no_intent': {
        'kind': 'eval', 'set': {'planner.intent_discount': 0.0}, 'predictor': 'flow',
        'label': 'w/o intent discount (no plan coordination)',
    },
    # --- active search -----------------------------------------------------------
    'no_search': {
        'kind': 'eval', 'set': {'planner.explore_weight': 0.0}, 'predictor': 'flow',
        'label': 'w/o active search (tracking reward only)',
    },
    'no_voronoi': {
        'kind': 'eval', 'set': {'planner.explore_voronoi': False}, 'predictor': 'flow',
        'label': 'w/o Voronoi split of the staleness map',
    },
    # --- train-time: stages 1-3 --------------------------------------------------
    'no_firsthand': {
        'kind': 'train', 'set': {'env.belief_drop': ['first_hand']}, 'predictor': 'flow',
        'label': 'w/o first-hand bit in the belief',
    },
    'no_age': {
        'kind': 'train', 'set': {'env.belief_drop': ['age']}, 'predictor': 'flow',
        'label': 'w/o staleness (age) column in the belief',
    },
    'no_consensus': {
        'kind': 'train', 'set': {'trajectory.consensus_weight': 0.0}, 'predictor': 'flow',
        'label': 'w/o consensus loss (prediction loss only)',
    },
}


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        node = config
        *parents, leaf = key.split('.')
        for parent in parents:
            node = node[parent]
        node[leaf] = copy.deepcopy(value)
    return config


def persistence(belief, n_agents, n_targets, horizon):
    """The believed position held over the roll, in map units, with its mask."""

    slots = belief.view(n_agents, n_targets, -1)
    base = slots[..., :2] * 1000.0
    predicted = base.unsqueeze(2).expand(-1, -1, horizon, -1).contiguous()
    return predicted, (slots[..., -1] > 0.5).float()


@torch.no_grad()
def score(checkpoint_path, variant, episodes, seed, device='cpu'):
    spec = VARIANTS[variant]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = copy.deepcopy(checkpoint['config'])
    config['train']['device'] = device
    config['env']['seed'] = seed
    config['env'].setdefault('belief_drop', [])
    # A train-time variant's overrides are already in its checkpoint's config;
    # applying them again is a no-op and keeps this function uniform.
    apply_overrides(config, spec['set'])

    env = build_env(config)
    env.update_statistics = False
    trajectory = build_trajectory(env_spec(env), config)
    trajectory.load_state_dict(checkpoint['trajectory'])
    trajectory.head.eval()
    planner = build_planner(config, device=device)

    per_episode = []
    acquisition, believed_all, union_all, overlaps = [], [], [], []
    for episode in range(episodes):
        # Episode k has to be the same episode for every variant, or the paired
        # delta carries the targets' randomness as noise.  Four generators are
        # involved and each is reset here: the layout (env reset seed), the
        # target agents (never seeded by MATE itself -- same workaround and same
        # reason as baseline.py), the flow head's samples (global torch), and
        # the MPPI candidate noise (the planner's own generator).
        episode_seed = seed + episode
        for offset, target_agent in enumerate(env.team_env.opponent_agents_ordered):
            target_agent._np_random, _ = seeding.np_random(
                episode_seed + 1000 * (offset + 1)
            )
        env._pending_seed = episode_seed
        torch.manual_seed(episode_seed)
        planner.generator.manual_seed(episode_seed)
        current = env.reset()
        planner.reset(env.n_agents, env.action_dim)
        tally = EpisodeTally()
        previous = None
        done = False
        while not done:
            belief = torch.as_tensor(current['belief'], dtype=torch.float32, device=device)
            known = current['belief'].reshape(env.n_agents, env.n_targets, -1)[..., -1] > 0.5
            if previous is not None:
                acquisition.append(float((known & ~previous).mean()))
            previous = known
            believed_all.append(float(known.mean()))
            union_all.append(float(known.any(axis=0).mean()))

            if spec['predictor'] == 'flow':
                predicted, mask = trajectory.predict(belief)
            else:
                predicted, mask = persistence(belief, env.n_agents, env.n_targets, planner.horizon)
            actions, intent = planner.plan(
                current['camera_states'], predicted, mask, current['peer_intent'],
                env.action_low, env.action_high, search=current.get('search'),
            )
            intent = planner.to_numpy(intent)
            overlaps.append(float(intent.sum(axis=0).max()))
            current, reward, done, info = env.step(planner.to_numpy(actions), intent)
            tally.add(reward, info)
        per_episode.append(tally.finish())
    env.close()

    coverage = [e['coverage_rate'] for e in per_episode]
    return {
        'variant': variant,
        'label': spec['label'],
        'kind': spec['kind'],
        'checkpoint': checkpoint_path,
        'train_seed': checkpoint['config']['env']['seed'],
        'scenario': config['env']['scenario'],
        'comm_range': config['env']['comm_range'],
        'eval_seed': seed,
        'episodes': episodes,
        'coverage_rate': float(np.mean(coverage)),
        'coverage_rate_std': float(np.std(coverage, ddof=1)) if episodes > 1 else 0.0,
        'raw_return': float(np.mean([e['raw_return'] for e in per_episode])),
        'acquisition': float(np.mean(acquisition)),
        'believed': float(np.mean(believed_all)),
        'union': float(np.mean(union_all)),
        'max_overlap': float(np.mean(overlaps)),
        'per_episode_coverage': coverage,
    }


# --------------------------------------------------------------------- report


def report(root, markdown_out=None):
    from scipy import stats

    rows = {}
    for path in sorted(glob.glob(os.path.join(root, '*', 'seed*.json'))):
        row = json.load(open(path))
        rows.setdefault(row['variant'], {})[row['train_seed']] = row
    if 'full' not in rows:
        raise SystemExit(f'no full-pipeline rows under {root}')
    full = rows['full']
    # Rows of variants no longer in the study (no_comm) are left on disk but
    # kept out of the table.
    for variant in sorted(set(rows) - set(VARIANTS)):
        print(f'skipping {variant}: not an ablation variant any more')
        del rows[variant]

    first = next(iter(full.values()))
    reach = first.get('comm_range')
    header = (
        f"{first['scenario']}, {first['episodes']} episodes per row, episode seed "
        f"{first['eval_seed']}. Channel: MATE RestrictedCommunicationRange around "
        f"MultiCamera, {f'range {reach:g}' if reach else 'unlimited range'}."
    )

    keys = ['coverage_rate', 'acquisition', 'union', 'max_overlap']
    lines = [
        '| Variant | Kind | Seeds | Coverage (mean ± std over seeds) | Δ vs full | p (paired t) '
        '| Acquisition | Union | Max overlap |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    order = [v for v in VARIANTS if v in rows]
    summary = {}
    for variant in order:
        by_seed = rows[variant]
        seeds = sorted(by_seed)
        mean = {k: float(np.mean([by_seed[s][k] for s in seeds])) for k in keys}
        spread = float(np.std([by_seed[s]['coverage_rate'] for s in seeds], ddof=1)) \
            if len(seeds) > 1 else 0.0

        paired = [s for s in seeds if s in full]
        delta, p = float('nan'), float('nan')
        if variant != 'full' and paired:
            a = np.array([by_seed[s]['coverage_rate'] for s in paired])
            b = np.array([full[s]['coverage_rate'] for s in paired])
            delta = float(np.mean(a - b))
            if len(paired) > 1 and np.any(a != b):
                p = float(stats.ttest_rel(a, b).pvalue)

        summary[variant] = {
            'label': VARIANTS.get(variant, {}).get('label', variant),
            'seeds': seeds, 'coverage_std_over_seeds': spread,
            'delta_vs_full': delta, 'p_paired': p, **mean,
        }
        label = VARIANTS.get(variant, {}).get('label', variant)
        kind = VARIANTS.get(variant, {}).get('kind', '?')
        lines.append(
            f'| {label} | {kind} | {len(seeds)} | {mean["coverage_rate"]:.4f} ± {spread:.4f} '
            f'| {"—" if variant == "full" else f"{delta:+.4f}"} '
            f'| {"—" if np.isnan(p) else f"{p:.3g}"} '
            f'| {mean["acquisition"]:.4f} | {mean["union"]:.3f} | {mean["max_overlap"]:.3f} |'
        )

    table = '\n'.join(lines)
    print(header + '\n')
    print(table)
    with open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=1)
    if markdown_out:
        with open(markdown_out, 'w', encoding='utf-8') as handle:
            handle.write(header + '\n\n' + table + '\n')


# ----------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('list')

    ev = sub.add_parser('eval')
    ev.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--episodes', type=int, default=50)
    ev.add_argument('--seed', type=int, default=12345, help='episode seed, shared by every row')
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--out', required=True)

    rp = sub.add_parser('report')
    rp.add_argument('--root', default='runs/ablation/eval')
    rp.add_argument('--markdown', default=None)

    args = parser.parse_args()
    if args.command == 'list':
        for name, spec in VARIANTS.items():
            sets = ' '.join(f'--set {k}={json.dumps(v)}' for k, v in spec['set'].items())
            print(f'{name:14s} {spec["kind"]:5s} {spec["predictor"]:7s} {sets}')
    elif args.command == 'eval':
        row = score(args.checkpoint, args.variant, args.episodes, args.seed, args.device)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(row, handle, indent=1)
        print(json.dumps({k: v for k, v in row.items() if k != 'per_episode_coverage'}))
    else:
        report(args.root, args.markdown)


if __name__ == '__main__':
    main()
