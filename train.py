"""Diffusion-driven trajectory planning on MATE: one pipeline, one learned module.

Every decision runs the same five stages, and only the third of them has weights:

    1  peer-to-peer round        each camera sends its target slots to the
                                 neighbours inside `comm_range` and merges what
                                 comes back into a belief          (MATEEnv)
    2  belief                    merged slots plus, per slot, whether it is first
                                 hand and how stale it is          (56 dims here)
    3  trajectory flow           conditional flow matching samples where every
                                 believed target goes over H steps (the only
                                 trained component)                (FlowTrajectoryHead)
    4  MPPI                      K candidate command sequences rolled through
                                 MATE's exact camera optics, scored by soft
                                 coverage of that sample, softmax-averaged
                                 (MPPIPlanner)
    5  intent                    the executed plan publishes what it means to
                                 cover, so neighbours discount it next decision

There is no actor and no critic: the policy is the planner, recomputed from
scratch every step.  The learning signal is dense and supervised -- true future
target positions out of the global state, which is training-time information a
camera never sees -- rather than a scalar team reward filtered through a policy
gradient.

A run writes everything a comparison needs to its own directory:

    config.yaml     the resolved configuration, including the seed
    metrics.jsonl   one line per iteration, written whether or not wandb is on
    checkpoint.pt   the latest weights
    best.pt         the weights at the best periodic evaluation
    results.json    the final evaluation, over more episodes than the periodic one

Nothing is resumed: runs are short, and a half-restored buffer would silently be
a different experiment.

Run:
    python train.py --seed 1 --tag v1 --steps 50000 --device cuda
    python train.py --seed 1 --tag smoke --steps 1000 --no-wandb --device cpu

``src/envs/config_resolver.py`` imports ``mate`` from the MATE-main checkout it
finds next to (or inside) the repository; ``MATE_ROOT`` overrides the search.
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

# The packages live under src/; the entrypoints stay at the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from algorithms.mppi_planner import MPPIPlanner
from algorithms.planning_collector import PlanningCollector
from algorithms.trajectory_buffer import TrajectoryBuffer
from envs.mate_wrapper import MATEEnv
from evaluate import evaluate_planner
from models.trajectory import TrajectoryFlowLearner


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'src', 'configs', 'default.yaml'
)
TERRAIN_SIZE = 1000.0


# ------------------------------------------------------------------------ config


def load_config(path, overrides=()):
    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    for override in overrides:
        key, _, raw = override.partition('=')
        if not _:
            raise ValueError(f'Override {override!r} is not of the form section.key=value')
        node = config
        *parents, leaf = key.split('.')
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise KeyError(f'Unknown config key {key!r}')
        node[leaf] = yaml.safe_load(raw)

    return config


def apply_arguments(config, args):
    """Command-line arguments win over the file, and are recorded in the run."""

    if args.seed is not None:
        config['env']['seed'] = args.seed
    if args.steps is not None:
        config['train']['total_env_steps'] = args.steps
    if args.device is not None:
        config['train']['device'] = args.device
    if args.wandb is not None:
        config['train']['wandb'] = args.wandb

    if config['train']['device'] == 'auto':
        config['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Pinned host memory is only meaningful for a device copy.
    config['train']['pin_memory'] = (
        config['train']['pin_memory'] and config['train']['device'].startswith('cuda')
    )

    if args.save_dir is not None:
        config['train']['save_dir'] = args.save_dir
    else:
        # One directory per (tag, seed) so a grid never collides and an
        # aggregation script can find every leaf by globbing.
        config['train']['save_dir'] = os.path.join(
            'runs', args.tag, f'seed{config["env"]["seed"]}'
        )
    config['train']['tag'] = args.tag
    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None, help='override total_env_steps')
    parser.add_argument('--device', type=str, default=None, help='cpu, cuda, cuda:0, auto')
    parser.add_argument('--tag', type=str, default='main', help='groups a grid of runs')
    parser.add_argument('--save-dir', type=str, default=None, help='overrides the run directory')
    parser.add_argument(
        '--final-episodes',
        type=int,
        default=50,
        help='episodes in the end-of-run evaluation written to results.json',
    )
    parser.add_argument('--wandb', dest='wandb', action='store_true', default=None)
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', default=None)
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='section.key=value',
        help='override a config leaf, repeatable',
    )
    return parser.parse_args()


# ------------------------------------------------------------------------ logging


class RunLogger:
    """Prints a line, appends the same numbers to ``metrics.jsonl``, feeds wandb.

    The file is what survives: a run that loses its terminal, or that was never
    attached to wandb because the cluster has no outbound network, still leaves
    every number it produced on disk.
    """

    def __init__(self, directory, config):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, 'metrics.jsonl')
        self.handle = open(self.path, 'a', encoding='utf-8')
        self.wandb = None

        if config['train']['wandb']:
            import wandb

            self.wandb = wandb.init(
                # WANDB_PROJECT overrides, so a run can be filed under a paper
                # name without touching the config.
                project=os.environ.get('WANDB_PROJECT', 'mate-diffusion-planning'),
                config=config,
                group=config['train']['tag'],
                name=f'seed{config["env"]["seed"]}',
                dir=directory,
            )

    def log(self, metrics, step):
        self.handle.write(json.dumps(metrics) + '\n')
        self.handle.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

        print(
            ' | '.join(
                f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                for k, v in metrics.items()
            ),
            flush=True,
        )

    def close(self):
        self.handle.close()
        if self.wandb is not None:
            self.wandb.finish()


# -------------------------------------------------------------------------- setup


def build_env(config, seed_offset=0):
    env_config = config['env']
    return MATEEnv(
        scenario=env_config['scenario'],
        max_episode_steps=env_config['max_episode_steps'],
        camera_comm=env_config['camera_comm'],
        comm_range=env_config['comm_range'],
        reward_scale=env_config['reward_scale'],
        frame_skip=env_config['frame_skip'],
        seed=env_config['seed'] + seed_offset,
        shared_fov=env_config['shared_fov'],
    )


def env_spec(env):
    return {
        'n_agents': env.n_agents,
        'n_targets': env.n_targets,
        'belief_dim': env.belief_dim,
        'obs_dim': env.obs_dim,
        'state_dim': env.state_dim,
        'action_dim': env.action_dim,
    }


def build_trajectory(spec, config):
    trajectory_config = config['trajectory']
    return TrajectoryFlowLearner(
        belief_dim=spec['belief_dim'],
        n_targets=spec['n_targets'],
        horizon=trajectory_config['horizon'],
        hidden_dim=trajectory_config['hidden_dim'],
        displacement_scale=trajectory_config['displacement_scale'],
        sample_steps=trajectory_config['sample_steps'],
        lr=trajectory_config['lr'],
        consensus_weight=trajectory_config['consensus_weight'],
        max_grad_norm=trajectory_config['max_grad_norm'],
        terrain_size=TERRAIN_SIZE,
        device=config['train']['device'],
    )


def build_planner(config, device=None):
    planner_config = config['planner']
    return MPPIPlanner(
        horizon=config['trajectory']['horizon'],
        samples=planner_config['samples'],
        temperature=planner_config['temperature'],
        noise_scale=planner_config['noise_scale'],
        discount=planner_config['discount'],
        range_softness=planner_config['range_softness'],
        angle_softness=planner_config['angle_softness'],
        intent_discount=planner_config['intent_discount'],
        # Read with defaults: a checkpoint written before active search existed
        # carries a planner block without these keys.
        recall_weight=planner_config.get('recall_weight', 0.0),
        recall_tau=planner_config.get('recall_tau', 5.0),
        recall_max_age=planner_config.get('recall_max_age', 8.0),
        recall_hypotheses=planner_config.get('recall_hypotheses', 4),
        recall_drift=planner_config.get('recall_drift', 17.0),
        explore_weight=planner_config.get('explore_weight', 0.0),
        explore_voronoi=planner_config.get('explore_voronoi', True),
        angle_weight=planner_config.get('angle_weight', 0.0),
        angle_threshold=planner_config.get('angle_threshold', 2.0),
        seed=config['env']['seed'],
        device=device or config['train']['device'],
    )


# -------------------------------------------------------------------------- main


def main():
    args = parse_args()
    config = apply_arguments(load_config(args.config, args.overrides), args)
    train_config = config['train']

    seed = config['env']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    directory = train_config['save_dir']
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'config.yaml'), 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    env = build_env(config)
    eval_env = build_env(config, seed_offset=10_000)
    # Evaluation reads the same normalization statistics the run collected under.
    eval_env.obs_rms = env.obs_rms
    eval_env.state_rms = env.state_rms

    spec = env_spec(env)
    trajectory = build_trajectory(spec, config)
    planner = build_planner(config)
    eval_planner = build_planner(config)
    buffer = TrajectoryBuffer(
        capacity=train_config['buffer_capacity'],
        n_agents=spec['n_agents'],
        n_targets=spec['n_targets'],
        belief_dim=spec['belief_dim'],
        pin_memory=train_config['pin_memory'],
    )
    collector = PlanningCollector(
        env, planner, trajectory, buffer, warmup_steps=train_config['warmup_steps']
    )

    logger = RunLogger(directory, config)

    # The device is printed because `auto` falls back to CPU silently when the
    # installed torch has no CUDA build -- on a GPU server that is worth seeing.
    device_name = train_config['device']
    if device_name.startswith('cuda'):
        device_name += f' ({torch.cuda.get_device_name(torch.device(device_name))})'

    print(env.describe(), flush=True)
    print(
        f'flow trajectory head, H={config["trajectory"]["horizon"]}, '
        f'{config["trajectory"]["sample_steps"]} Euler steps | '
        f'MPPI {config["planner"]["samples"]} candidates | '
        f'seed {seed} | device {device_name} | pin_memory {train_config["pin_memory"]}',
        flush=True,
    )
    print(f'run directory: {directory}', flush=True)

    env_steps = 0
    iteration = 0
    best_coverage = float('-inf')
    horizon = config['trajectory']['horizon']
    start = time.time()

    while env_steps < train_config['total_env_steps']:
        iteration += 1
        env_steps += collector.collect(train_config['env_steps_per_iter'])

        trajectory_metrics = defaultdict(float)
        updates = train_config['trajectory_updates_per_iter']
        performed = 0
        for _ in range(updates):
            window = buffer.sample_windows(
                train_config['trajectory_batch_size'], horizon, train_config['device']
            )
            if window is None:
                break
            beliefs, future = window
            for key, value in trajectory.update(beliefs, future).items():
                trajectory_metrics[key] += value
            performed += 1
        for key in trajectory_metrics:
            trajectory_metrics[key] /= max(performed, 1)

        elapsed = time.time() - start
        metrics = {
            'env_steps': env_steps,
            'iteration': iteration,
            'system/sps': env_steps / max(elapsed, 1e-6),
            'system/elapsed_hours': elapsed / 3600.0,
            'system/buffer': len(buffer),
            **trajectory_metrics,
            **collector.drain_stats(),
        }

        if iteration % train_config['eval_every'] == 0:
            metrics.update(
                evaluate_planner(
                    eval_env, eval_planner, trajectory, train_config['eval_episodes']
                )
            )
            checkpoint = {
                'env_steps': env_steps,
                'trajectory': trajectory.state_dict(),
                'obs_rms': env.obs_rms.state_dict(),
                'state_rms': env.state_rms.state_dict(),
                'config': copy.deepcopy(config),
            }
            torch.save(checkpoint, os.path.join(directory, 'checkpoint.pt'))
            if metrics['eval/coverage_rate'] > best_coverage:
                best_coverage = metrics['eval/coverage_rate']
                torch.save(checkpoint, os.path.join(directory, 'best.pt'))

        logger.log(metrics, step=env_steps)

    # The number the comparison is made on: more episodes than the periodic
    # evaluation, so the standard error is small enough to separate conditions.
    final = evaluate_planner(eval_env, eval_planner, trajectory, args.final_episodes)
    # The periodic branch above is the only other writer, so a run whose
    # `eval_every` never came up would otherwise finish with a results.json and
    # no weights at all.  Writing here costs one save and makes every run
    # recoverable.
    torch.save(
        {
            'env_steps': env_steps,
            'trajectory': trajectory.state_dict(),
            'obs_rms': env.obs_rms.state_dict(),
            'state_rms': env.state_rms.state_dict(),
            'config': copy.deepcopy(config),
        },
        os.path.join(directory, 'checkpoint.pt'),
    )

    results = {
        'seed': seed,
        'tag': train_config['tag'],
        'scenario': config['env']['scenario'],
        'env_steps': env_steps,
        'episodes': args.final_episodes,
        'coverage_rate': final['eval/coverage_rate'],
        'coverage_rate_std': final['eval/coverage_rate_std'],
        'episode_return': final['eval/episode_return'],
        'believed_fraction': final['eval/believed_fraction'],
        'best_periodic_coverage': best_coverage if best_coverage > float('-inf') else None,
        'wall_clock_hours': (time.time() - start) / 3600.0,
        'directory': directory,
    }
    with open(os.path.join(directory, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)

    if logger.wandb is not None:
        logger.wandb.summary.update(results)

    print(
        f'\nfinal: coverage {results["coverage_rate"]:.4f} '
        f'+- {results["coverage_rate_std"]:.4f} over {args.final_episodes} episodes '
        f'({env_steps} env steps, {results["wall_clock_hours"]:.2f} h)',
        flush=True,
    )

    logger.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
