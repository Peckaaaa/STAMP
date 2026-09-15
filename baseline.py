"""Every method on this task, scored under one protocol.

One command, one table.  The rule agents MATE ships, and a trained checkpoint if
one is given, all replay the *same* episode seeds against the same opponents for
the same episode length, under the coverage-rate definition the training loop
logs -- so the rows can be read against each other, and against
``eval/coverage_rate``, with no footnote.

    python baseline.py --episodes 20
    python baseline.py --episodes 20 --checkpoint runs/v1/seed1/best.pt
    python baseline.py --episodes 20 --matrix       # the fusion cross, below

Three questions the table answers:

  floor    what do the rule agents score, and how much of that is coordination
           rather than looking?  ``random`` is the floor; ``greedy`` is the bar
           a learned policy has to clear, and it is a high one.
  ceiling  ``--matrix`` re-runs every agent with and without MATE's message
           round and with and without oracle field-of-view fusion, and prints
           the paired deltas.  No learned peer-to-peer scheme can beat an oracle
           that hands every camera the union of the team's view for free, so the
           oracle column is the go/no-go for the belief-fusion blocks.
  where    a checkpoint sits between them.  When its planner carries search
           terms it is scored twice -- as trained, and with the terms switched
           off -- which is the ablation, not a second run: the terms live in the
           reward rather than in any weights.

MAPPO and MAPPO+HRL are **not** run here.  Reproducing them needs an
actor-critic and a hierarchical action wrapper that neither this repository nor
``MATE-main`` ships, trained for the 10M environment steps the MATE paper used
-- roughly forty hours at this machine's throughput.  Their published numbers
are printed under the table, marked as what they are: read off a figure in
someone else's paper, not measured by this script.

Every configuration replays the same episode seeds, so a delta between two rows
is a paired comparison on identical layouts rather than two independent samples.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from envs.config_resolver import ensure_mate_importable, resolve_scenario
from envs.observation_fusion import fuse_camera_observations, target_visibility


AGENTS = ('random', 'naive', 'greedy', 'heuristic')

# HeuristicCameraAgent computes its joint goal assignment inside the message
# round (``send_responses``), so ``act`` raises without it: its coordination and
# its communication are the same mechanism and cannot be measured apart.
REQUIRES_COMMUNICATION = ('heuristic',)


def build_camera_agents(name, num_cameras, seed, memory_period=None):
    from mate.agents import (
        GreedyCameraAgent,
        HeuristicCameraAgent,
        NaiveCameraAgent,
        RandomCameraAgent,
    )

    factory = {
        'random': RandomCameraAgent,
        'naive': NaiveCameraAgent,
        'greedy': GreedyCameraAgent,
        'heuristic': HeuristicCameraAgent,
    }[name]

    # GreedyCameraAgent keeps seeing a target for ``memory_period`` steps after
    # it leaves the field of view.  Shrinking that window is how much of its
    # coverage rests on remembering rather than on looking, which is the
    # question a memoryless single-step actor runs into.
    keywords = {}
    if memory_period is not None and name == 'greedy':
        keywords['memory_period'] = memory_period

    prototype = factory(**keywords)
    prototype.seed(seed)
    return prototype.spawn(num_cameras)


def run_rule_agent(
    scenario,
    agent_name,
    episodes,
    max_episode_steps,
    base_seed,
    shared_fov,
    communicate,
    memory_period=None,
    action_noise=0.0,
):
    """Whole episodes of a built-in camera agent against MATE's greedy targets.

    The targets are ``GreedyTargetAgent`` -- the same opponent the world model is
    trained against -- so the numbers are directly comparable with
    ``eval/coverage_rate``.

    ``communicate`` drives MATE's own agent-to-agent message round.  It has to be
    switchable: ``GreedyCameraAgent`` already ships a hand-written version of the
    proposed pipeline -- it sends observed target states to teammates in range and
    keeps a memory of what it was told -- so with that round left on, an oracle
    field of view adds almost nothing and the measurement says nothing about what
    fusion is worth.  Turning it off isolates the information sharing.
    """

    import mate
    from gymnasium.utils import seeding
    from mate.agents import GreedyTargetAgent

    base_env = mate.make_environment(config=scenario, max_episode_steps=max_episode_steps)
    env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())
    unwrapped = env.unwrapped
    counts = (unwrapped.num_cameras, unwrapped.num_targets, unwrapped.num_obstacles)

    # Noise is added in the wrapper's [-1, 1] units, so a sigma here is directly
    # comparable with the residual error of a learned policy measured there.
    low = unwrapped.camera_action_space.low.astype(np.float64)
    high = unwrapped.camera_action_space.high.astype(np.float64)
    noise_generator = np.random.default_rng(base_seed)

    episode_coverage, episode_real_coverage = [], []
    per_camera_visibility, union_visibility = [], []

    for episode in range(episodes):
        agents = build_camera_agents(
            agent_name, counts[0], base_seed + episode, memory_period
        )
        # Before ``reset``: the target agents carry their own generators, the
        # opponent prototype is never seeded by the environment, and
        # ``reset`` already draws on them.  Seeding here is what makes two
        # settings replay the same opponent behaviour and not merely the same
        # starting layout; without it the paired delta carries the targets'
        # randomness as noise.
        for offset, target_agent in enumerate(env.opponent_agents_ordered):
            # Not ``target_agent.seed()``: that path ends in
            # ``action_space.seed(self.np_random.integers(...))``, and this
            # gymnasium rejects the numpy integer it hands over -- the same kind
            # of breakage as ``env.seed()``.  Setting the generator is the only
            # working route, and it is the one the agent's decisions read.
            target_agent._np_random, _ = seeding.np_random(
                base_seed + episode + 1000 * (offset + 1)
            )
        observation, _ = env.reset(seed=base_seed + episode)
        if shared_fov:
            observation = fuse_camera_observations(observation, *counts)
        mate.group_reset(agents, observation)
        infos = None

        coverage, real_coverage = [], []
        done = False
        while not done:
            visible = target_visibility(observation, *counts)
            per_camera_visibility.append(visible.mean())
            union_visibility.append(visible.any(axis=0).mean())

            if communicate:
                joint_action = mate.group_step(unwrapped, agents, observation, infos)
            else:
                mate.group_observe(agents, observation, infos)
                joint_action = mate.group_act(agents, observation, infos)
            if action_noise > 0.0:
                normalized = 2.0 * (np.asarray(joint_action) - low) / (high - low) - 1.0
                normalized = np.clip(
                    normalized + noise_generator.normal(0.0, action_noise, normalized.shape),
                    -1.0,
                    1.0,
                )
                joint_action = low + 0.5 * (normalized + 1.0) * (high - low)

            observation, _, terminated, truncated, infos = env.step(joint_action)
            if shared_fov:
                observation = fuse_camera_observations(observation, *counts)
            done = bool(terminated) or bool(truncated)

            coverage.append(float(infos[0]['coverage_rate']))
            real_coverage.append(float(infos[0]['real_coverage_rate']))

        episode_coverage.append(float(np.mean(coverage)))
        episode_real_coverage.append(float(np.mean(real_coverage)))

    env.close()
    return {
        'coverage_rate': float(np.mean(episode_coverage)),
        'coverage_rate_std': float(np.std(episode_coverage)),
        'real_coverage_rate': float(np.mean(episode_real_coverage)),
        'per_camera_visibility': float(np.mean(per_camera_visibility)),
        'union_visibility': float(np.mean(union_visibility)),
        'episodes': episode_coverage,
    }


def run_checkpoint(path, device, episodes, base_seed, shared_fov,
                   scenario=None, overrides=None):
    """A trained run under the same protocol as the rule agents.

    A checkpoint holds the trajectory head and the configuration it was trained
    with; the planner is rebuilt from that configuration, since it carries no
    weights of its own.
    """

    import torch

    from train import build_planner, build_trajectory, env_spec
    from envs.mate_wrapper import MATEEnv
    from evaluate import evaluate_planner

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config['train']['device'] = device

    env = MATEEnv(
        scenario=scenario or config['env']['scenario'],
        max_episode_steps=config['env']['max_episode_steps'],
        camera_comm=config['env']['camera_comm'],
        comm_range=config['env']['comm_range'],
        reward_scale=config['env']['reward_scale'],
        frame_skip=config['env']['frame_skip'],
        seed=base_seed,
        shared_fov=shared_fov,
    )
    env.obs_rms.load_state_dict(checkpoint['obs_rms'])
    env.state_rms.load_state_dict(checkpoint['state_rms'])

    trajectory = build_trajectory(env_spec(env), config)
    trajectory.load_state_dict(checkpoint['trajectory'])
    trajectory.head.eval()
    if overrides:
        config['planner'].update(overrides)
    planner = build_planner(config, device=device)

    metrics = evaluate_planner(env, planner, trajectory, episodes)
    env.close()
    return {
        'coverage_rate': metrics['eval/coverage_rate'],
        'coverage_rate_std': metrics['eval/coverage_rate_std'],
        'real_coverage_rate': float('nan'),
        'per_camera_visibility': float('nan'),
        'union_visibility': metrics['eval/believed_fraction'],
        'episodes': [],
    }


SETTINGS = ('nocomm', 'nocomm+fov', 'comm', 'comm+fov')

#: Published elsewhere, printed for orientation, never measured by this script.
#: The MATE paper's Figure 4a, 4C vs. 8T (9O), at 10M environment steps; the
#: model-free MAPPO row is this repository's own earlier run at 150k steps.
PUBLISHED = (
    ('MAPPO + HRL', 0.55, 'MATE paper, Fig. 4a, 10M steps'),
    ('IPPO + HRL', 0.52, 'MATE paper, Fig. 4a, 10M steps'),
    ('MAPPO, model-free', 0.3405, 'this repository, 150k steps'),
)


def paired_delta(row, better, worse):
    """Mean per-episode difference; the two settings replayed the same seeds."""

    if not row.get(better, {}).get('episodes') or not row.get(worse, {}).get('episodes'):
        return float('nan'), float('nan')
    differences = np.array(row[better]['episodes']) - np.array(row[worse]['episodes'])
    return float(differences.mean()), float(differences.std())


def format_table(rows, setting):
    """The comparison as markdown, weakest policy first."""

    def cell(value, places=4):
        return '--' if value != value else f'{value:.{places}f}'

    lines = [
        '| Policy | Coverage | +- | Per-camera view | Team union |',
        '|---|---|---|---|---|',
    ]
    scored = [(row, row.get(setting)) for row in rows]
    scored = [(row, result) for row, result in scored if result is not None]
    scored.sort(key=lambda pair: pair[1]['coverage_rate'])

    for row, result in scored:
        lines.append(
            f"| {row['policy']} | {cell(result['coverage_rate'])} "
            f"| {cell(result['coverage_rate_std'])} "
            f"| {cell(result['per_camera_visibility'], 3)} "
            f"| {cell(result['union_visibility'], 3)} |"
        )
    return '\n'.join(lines)


def report_matrix(row):
    """The fusion cross for one agent, with the paired deltas underneath."""

    print()
    print(row['policy'])
    for setting in SETTINGS:
        result = row.get(setting)
        if result is None:
            continue
        print(
            f"  {setting:>11}: coverage {result['coverage_rate']:.4f} "
            f"+- {result['coverage_rate_std']:.4f} | real {result['real_coverage_rate']:.4f}"
        )

    if 'nocomm' in row:
        visibility = row['nocomm']
        print(
            f"  {'visibility':>11}: per camera {visibility['per_camera_visibility']:.3f}, "
            f"team union {visibility['union_visibility']:.3f}"
        )

    for label, better, worse in (
        ('oracle fusion', 'nocomm+fov', 'nocomm'),
        ('MATE protocol', 'comm', 'nocomm'),
        ('fusion on top', 'comm+fov', 'comm'),
    ):
        mean, std = paired_delta(row, better, worse)
        if mean == mean:  # not NaN
            print(f"  {label:>11}: {mean:+.4f} +- {std:.4f} coverage (paired)")


def write_results(args, scenario, rows):
    """Persist after every agent, merging into whatever the file already holds.

    A sweep is long enough that losing the finished agents to a crash in a later
    one is not acceptable, and re-running a single agent has to be able to fill
    its row in without discarding the rest.
    """

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    merged = {}
    if os.path.isfile(args.out):
        with open(args.out, 'r', encoding='utf-8') as handle:
            existing = json.load(handle)
        if (
            existing.get('scenario') == scenario
            and existing.get('episodes') == args.episodes
            and existing.get('seed') == args.seed
        ):
            merged = {row['policy']: row for row in existing.get('rows', [])}

    merged.update({row['policy']: row for row in rows})

    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'scenario': scenario,
                'episodes': args.episodes,
                'max_episode_steps': args.max_episode_steps,
                'seed': args.seed,
                'rows': list(merged.values()),
            },
            handle,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='MATE-4v8-9')
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--agents', type=str, nargs='*', default=list(AGENTS), choices=AGENTS)
    parser.add_argument(
        '--matrix',
        action='store_true',
        help='run every agent under all four communication/fusion settings and '
             'report the paired deltas, instead of one table under --setting',
    )
    parser.add_argument(
        '--setting',
        type=str,
        default='comm',
        choices=SETTINGS,
        help="which setting the table is scored under (default: MATE's own "
             'message round, no oracle)',
    )
    parser.add_argument(
        '--greedy-memory',
        type=int,
        default=None,
        help="GreedyCameraAgent's memory window in steps (MATE's default is 25)",
    )
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument(
        '--no-ablation',
        action='store_true',
        help='score a checkpoint only as trained, skipping the search-off row',
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='runs/baseline/results.json')
    return parser.parse_args()


def checkpoint_rows(args, settings):
    """A trained run, and the same weights with the search terms switched off.

    The ablation is free: the search terms are part of the planner's reward
    rather than of any weights, so switching them off is a second score of the
    same checkpoint rather than a second run.
    """

    import torch

    configuration = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False
    )['config']
    searching = any(
        configuration['planner'].get(key)
        for key in ('explore_weight', 'recall_weight', 'angle_weight')
    )

    variants = [('planning + search' if searching else 'planning', None)]
    if searching and not args.no_ablation:
        variants.append(
            (
                'planning, tracking only',
                {'explore_weight': 0.0, 'recall_weight': 0.0, 'angle_weight': 0.0},
            )
        )

    rows = []
    for label, overrides in variants:
        row = {'policy': label}
        for setting in settings:
            if setting.startswith('nocomm'):
                # The wrapper's channel is what builds the belief the planner
                # reads, so scoring it with the channel off would be a different
                # pipeline rather than a different setting.
                continue
            row[setting] = run_checkpoint(
                args.checkpoint,
                args.device,
                args.episodes,
                args.seed,
                shared_fov=setting.endswith('fov'),
                scenario=args.scenario,
                overrides=overrides,
            )
            print(
                f"  {label:>24} / {setting:<10} "
                f"coverage {row[setting]['coverage_rate']:.4f}",
                flush=True,
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    ensure_mate_importable()
    scenario = resolve_scenario(args.scenario)
    settings = SETTINGS if args.matrix else (args.setting,)

    print(
        f'{scenario} | {args.episodes} episodes x {args.max_episode_steps} steps '
        f'| seed {args.seed} | setting {"matrix" if args.matrix else args.setting}'
    )

    rows = []
    for name in args.agents:
        label = name
        if args.greedy_memory is not None and name == 'greedy':
            label = f'{name} (memory {args.greedy_memory})'
        row = {'policy': label}
        for setting in settings:
            communicate = setting.startswith('comm')
            shared_fov = setting.endswith('fov')
            if not communicate and name in REQUIRES_COMMUNICATION:
                continue
            row[setting] = run_rule_agent(
                scenario,
                name,
                args.episodes,
                args.max_episode_steps,
                args.seed,
                shared_fov=shared_fov,
                communicate=communicate,
                memory_period=args.greedy_memory,
            )
            print(
                f'  {label:>24} / {setting:<10} '
                f"coverage {row[setting]['coverage_rate']:.4f}",
                flush=True,
            )
        rows.append(row)
        write_results(args, scenario, rows)

    if args.checkpoint:
        rows.extend(checkpoint_rows(args, settings))
        write_results(args, scenario, rows)

    if args.matrix:
        for row in rows:
            report_matrix(row)
    else:
        print()
        print(format_table(rows, args.setting))

    print()
    print('Published elsewhere, not measured here:')
    for label, coverage, source in PUBLISHED:
        print(f'  {label:<20} {coverage:.4f}   {source}')

    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
