"""Score a checkpoint under a given active-search configuration.

The search terms live in the planner's reward, not in the weights, so a
configuration can be measured on a checkpoint that was trained without them --
which is the whole reason they were put there.  Coverage is the number to
report, but it sits at the end of the causal chain and its episode spread is
large; ``acquisition`` is the thing a search term acts on directly:

    acquisition   slot-observations per decision that went unknown -> known
    believed      fraction of slots a camera knows about
    union         fraction the team knows about between them

Usage:
    python search_eval.py --checkpoint runs/v1/seed2/best.pt --episodes 20 \
        --set planner.explore_weight=0.3
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')

from train import build_planner, build_trajectory, env_spec   # noqa: E402
from envs.mate_wrapper import MATEEnv                          # noqa: E402


def parse_overrides(pairs):
    out = {}
    for item in pairs or []:
        key, _, value = item.partition('=')
        section, _, leaf = key.strip().partition('.')
        if section != 'planner':
            raise SystemExit(f'only planner.* overrides are supported, got {key!r}')
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        out[leaf] = parsed
    return out


def run(path, episodes, device, seed, overrides, label, predictor='flow'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config['train']['device'] = device
    config['planner'].update(overrides)

    env = MATEEnv(
        scenario=config['env']['scenario'],
        max_episode_steps=config['env']['max_episode_steps'],
        camera_comm=config['env']['camera_comm'],
        comm_range=config['env']['comm_range'],
        reward_scale=config['env']['reward_scale'],
        frame_skip=config['env']['frame_skip'],
        seed=seed,
        shared_fov=False,
    )
    env.obs_rms.load_state_dict(checkpoint['obs_rms'])
    env.state_rms.load_state_dict(checkpoint['state_rms'])

    trajectory = build_trajectory(env_spec(env), config)
    trajectory.load_state_dict(checkpoint['trajectory'])
    trajectory.head.eval()
    planner = build_planner(config, device=device)

    coverages, acquisitions, believed_all, union_all, overlaps = [], [], [], [], []
    for _ in range(episodes):
        current = env.reset()
        planner.reset(env.n_agents, env.action_dim)
        previous_known = None
        done, per_step = False, []
        while not done:
            belief = torch.as_tensor(current['belief'], dtype=torch.float32, device=device)
            slots = current['belief'].reshape(env.n_agents, env.n_targets, -1)
            known = slots[..., -1] > 0.5
            if previous_known is not None:
                acquisitions.append(float((known & ~previous_known).mean()))
            previous_known = known
            believed_all.append(float(known.mean()))
            union_all.append(float(known.any(axis=0).mean()))

            if predictor == 'flow':
                predicted, mask = trajectory.predict(belief)
            else:
                # "Target stands still": the believed position, held over the
                # roll.  Measured to score the same as the flow head, so it is
                # the honest control for anything the head is credited with.
                base = belief.view(env.n_agents, env.n_targets, -1)[..., :2] * 1000.0
                predicted = base.unsqueeze(2).expand(-1, -1, planner.horizon, -1).contiguous()
                mask = torch.as_tensor(known.astype('float32'), device=device)
            actions, intent = planner.plan(
                current['camera_states'], predicted, mask, current['peer_intent'],
                env.action_low, env.action_high, search=current.get('search'),
            )
            intent = planner.to_numpy(intent)
            overlaps.append(float(intent.sum(axis=0).max()))
            current, _, done, info = env.step(planner.to_numpy(actions), intent)
            per_step.append((info['coverage_rate'], info['env_steps']))

        total = sum(steps for _, steps in per_step)
        coverages.append(sum(rate * steps for rate, steps in per_step) / max(total, 1))
    env.close()

    return {
        'label': label,
        'predictor': predictor,
        'overrides': overrides,
        'episodes': episodes,
        'coverage': float(np.mean(coverages)),
        'coverage_std': float(np.std(coverages, ddof=1)),
        'acquisition': float(np.mean(acquisitions)),
        'believed': float(np.mean(believed_all)),
        'union': float(np.mean(union_all)),
        'max_overlap': float(np.mean(overlaps)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--label', default='run')
    parser.add_argument('--predictor', default='flow', choices=['flow', 'persist'])
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    row = run(args.checkpoint, args.episodes, args.device, args.seed,
              parse_overrides(args.overrides), args.label, args.predictor)
    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(row, handle, indent=1)


if __name__ == '__main__':
    main()
