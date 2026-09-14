"""Closed-loop shape and device check for the unified pipeline.

Runs one decision and one training step and asserts that every tensor lives on
the requested device and carries the layout the next stage expects.  It is the
cheapest way to find out that a GPU box is misconfigured, or that an axis got
transposed, without waiting for a training run to crash an hour in.

The axis worth checking is the trajectory label: the buffer stores positions
time-major and the head predicts target-major, so a window has to come back as
``(B, n_targets, horizon, 2)``.  Getting that backwards is silent -- both are
four-dimensional float tensors -- and it cost a debugging session once already.

    python check_pipeline.py --device cpu
    python check_pipeline.py --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from algorithms.trajectory_buffer import TrajectoryBuffer
from train import build_env, build_planner, build_trajectory, env_spec, load_config, DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--steps', type=int, default=60, help='transitions to fill the buffer')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda but torch reports no CUDA build or no visible GPU')

    config = load_config(args.config)
    config['train']['device'] = str(device)
    config['train']['pin_memory'] = device.type == 'cuda'

    env = build_env(config)
    spec = env_spec(env)
    trajectory = build_trajectory(spec, config)
    planner = build_planner(config)

    n, targets, horizon = spec['n_agents'], spec['n_targets'], config['trajectory']['horizon']
    print(env.describe())
    print(
        f'expected: belief ({n}, {spec["belief_dim"]}) | '
        f'trajectory ({n}, {targets}, {horizon}, 2) | actions ({n}, 2)'
    )

    # ------------------------------------------------------------ one decision
    current = env.reset()
    beliefs = torch.as_tensor(current['belief'], dtype=torch.float32, device=device)
    predicted, mask = trajectory.predict(beliefs)
    actions, intent = planner.plan(
        current['camera_states'], predicted, mask, current['peer_intent'],
        env.action_low, env.action_high,
    )

    checks = {
        'belief': (beliefs, (n, spec['belief_dim'])),
        'trajectory': (predicted, (n, targets, horizon, 2)),
        'believed': (mask, (n, targets)),
        'actions': (actions, (n, 2)),
        'intent': (intent, (n, targets)),
    }
    for name, (tensor, shape) in checks.items():
        assert tuple(tensor.shape) == shape, f'{name}: {tuple(tensor.shape)} != {shape}'
        assert tensor.device.type == device.type, f'{name} on {tensor.device}, expected {device}'
        print(f'  {name:<11} {tuple(tensor.shape)}  {tensor.device}')

    assert float(actions.abs().max()) <= 1.0 + 1e-5, 'actions must stay in the normalized box'

    # ------------------------------------------------------- one training step
    buffer = TrajectoryBuffer(
        capacity=max(args.steps * 2, 256),
        n_agents=n,
        n_targets=targets,
        belief_dim=spec['belief_dim'],
        pin_memory=config['train']['pin_memory'],
    )
    for _ in range(args.steps):
        command = np.random.uniform(-1.0, 1.0, (n, env.action_dim)).astype(np.float32)
        nxt, _, done, _ = env.step(command)
        buffer.add(current['belief'], current['target_positions'], done)
        current = env.reset() if done else nxt

    window = buffer.sample_windows(args.batch_size, horizon, device)
    assert window is not None, 'buffer produced no window -- raise --steps'
    belief_batch, future = window

    assert tuple(future.shape[1:]) == (targets, horizon, 2), (
        f'labels are {tuple(future.shape)}, expected (B, {targets}, {horizon}, 2) -- '
        'target-major, not time-major'
    )
    assert belief_batch.device.type == device.type and future.device.type == device.type
    print(f'  window      beliefs {tuple(belief_batch.shape)} | labels {tuple(future.shape)}')

    metrics = trajectory.update(belief_batch, future)
    for key, value in metrics.items():
        assert np.isfinite(value), f'{key} is {value}'
    print('  update      ' + ' '.join(f'{k.split("/")[-1]}={v:.4f}' for k, v in metrics.items()))

    if device.type == 'cuda':
        print(f'  gpu memory  {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB peak')

    print(f'OK on {device}')
    env.close()


if __name__ == '__main__':
    main()
