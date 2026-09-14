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


@torch.no_grad()
def evaluate_planner(env, planner, trajectory, episodes):
    """Whole episodes driven by online planning.  Nothing is trained here.

    The planner is stateful within an episode -- it warm-starts each decision
    from the plan it made at the last one -- so its warm start is cleared at
    every episode boundary.  The trajectory head still *samples*, so this is a
    stochastic policy by construction; the spread across episodes is reported
    beside the mean rather than hidden by a deterministic mode.
    """

    coverages, returns, believed = [], [], []
    for _ in range(episodes):
        current = env.reset()
        planner.reset(env.n_agents, env.action_dim)

        done = False
        total, coverage = 0.0, []
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
            total += reward
            coverage.append((info['coverage_rate'], info['env_steps']))

        returns.append(total)
        coverages.append(weighted_mean(coverage))

    return {
        'eval/episode_return': float(np.mean(returns)),
        'eval/coverage_rate': float(np.mean(coverages)),
        'eval/coverage_rate_std': float(np.std(coverages)),
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
