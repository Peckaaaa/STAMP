"""Coverage-rate evaluation, standalone or called from training.

    python evaluate.py --checkpoint runs/v1/seed1/best.pt --episodes 50
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def weighted_mean(pairs):
    """Mean of ``(value, weight)`` pairs; 0.0 when there is no weight.

    Coverage rate is a mean over MATE steps, so a decision that held its action
    for five of them counts five times as much as one cut short by an episode end.
    """

    total = sum(weight for _, weight in pairs)
    if total == 0:
        return 0.0
    return float(sum(value * weight for value, weight in pairs) / total)


class EpisodeTally:
    """Per-episode totals, shared by the training collector and evaluation.

    Two rewards are tracked side by side.  ``return`` / ``reward`` are the
    training signal -- coverage rate times ``reward_scale``.  ``raw_*`` is MATE's
    own camera-team reward: +1 per step for every loaded target a camera tracks,
    minus the freight and bounty of every cargo the targets deliver.  The game
    is zero-sum, so the camera team wins an episode when that sum is positive.
    """

    def __init__(self):
        self.returns = 0.0
        self.raw_return = 0.0
        self.steps = 0
        self.rewards = []
        self.coverage = []

    def add(self, reward, info):
        self.returns += reward
        self.raw_return += info['raw_reward']
        self.steps += info['env_steps']
        # A decision's reward is already its mean over the MATE steps it held,
        # so the per-step reward is weighted the same way coverage is.
        self.rewards.append((reward, info['env_steps']))
        self.coverage.append((info['coverage_rate'], info['env_steps']))

    def finish(self):
        steps = max(self.steps, 1)
        return {
            'return': self.returns,
            'reward': weighted_mean(self.rewards),
            'raw_return': self.raw_return,
            'raw_reward': self.raw_return / steps,
            'win': float(self.raw_return > 0.0),
            'coverage_rate': weighted_mean(self.coverage),
        }


def summarize_episodes(episodes, prefix):
    """Means over finished episodes, under ``prefix/``."""

    def mean(key):
        return float(np.mean([e[key] for e in episodes]))

    coverages = [e['coverage_rate'] for e in episodes]
    return {
        f'{prefix}/episode_return': mean('return'),
        f'{prefix}/reward': mean('reward'),
        f'{prefix}/raw_return': mean('raw_return'),
        f'{prefix}/raw_reward': mean('raw_reward'),
        f'{prefix}/win_rate': mean('win'),
        f'{prefix}/coverage_rate': float(np.mean(coverages)),
        f'{prefix}/coverage_rate_std': float(np.std(coverages)),
    }


@torch.no_grad()
def evaluate_planner(env, planner, trajectory, episodes):
    """Whole episodes driven by online planning.  Nothing is trained here.

    The planner is stateful within an episode -- it warm-starts each decision
    from the plan it made at the last one -- so its warm start is cleared at
    every episode boundary.  The trajectory head still *samples*, so this is a
    stochastic policy by construction; the spread across episodes is reported
    beside the mean rather than hidden by a deterministic mode.
    """

    finished, believed = [], []
    for _ in range(episodes):
        current = env.reset()
        planner.reset(env.n_agents, env.action_dim)

        done = False
        tally = EpisodeTally()
        while not done:
            beliefs = torch.as_tensor(
                current['belief'], dtype=torch.float32, device=trajectory.device
            )
            predicted, mask = trajectory.predict(beliefs)
            actions, intent = planner.plan(
                current['camera_states'],
                predicted,
                mask,
                current['peer_intent'],
                env.action_low,
                env.action_high,
                search=current.get('search'),
            )
            believed.append(float(mask.mean().item()))
            current, reward, done, info = env.step(
                planner.to_numpy(actions), planner.to_numpy(intent)
            )
            tally.add(reward, info)

        finished.append(tally.finish())

    return {
        **summarize_episodes(finished, 'eval'),
        'eval/believed_fraction': float(np.mean(believed)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--scenario', type=str, default=None, help='override the trained scenario')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main():
    # Imported here, not at module scope: train.py imports from this module, and
    # a top-level import back into train.py would be circular.
    from train import build_env, build_planner, build_trajectory, env_spec

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint['config']
    config['train']['device'] = args.device
    config['env']['seed'] = args.seed
    if args.scenario is not None:
        config['env']['scenario'] = args.scenario

    env = build_env(config)
    trajectory = build_trajectory(env_spec(env), config)
    trajectory.load_state_dict(checkpoint['trajectory'])
    trajectory.head.eval()
    planner = build_planner(config, device=args.device)

    print(env.describe())
    metrics = evaluate_planner(env, planner, trajectory, args.episodes)
    for key, value in metrics.items():
        print(f'{key}={value:.4f}')
    env.close()


if __name__ == '__main__':
    main()
