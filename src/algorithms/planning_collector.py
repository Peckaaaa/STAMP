"""Rollouts driven by online planning instead of a policy.

Every decision is planned from scratch against a fresh sample from the
trajectory head, so the collected data is used for exactly one thing: fitting
that head.  The interesting consequence is where the learning signal now lives.
A policy gradient has to discover, through a scalar team reward, both what the
targets do and what to do about it.  Here the second half is arithmetic the
planner redoes every step, and only the first half is learned -- from a dense
supervised signal that does not depend on the policy having been any good.

The environment is numpy and the planner is torch, so the boundary is crossed
once per decision, on arrays of a few hundred floats.  Everything downstream of
the belief -- sampling the flow, rolling 64 candidates, scoring coverage --
stays on the device.
"""

import numpy as np
import torch

from evaluate import weighted_mean


class PlanningCollector:
    """Steps the real environment with an MPPI plan at every decision."""

    def __init__(self, env, planner, trajectory, buffer, warmup_steps=0):
        self.env = env
        self.planner = planner
        self.trajectory = trajectory
        self.buffer = buffer
        # Until the head has seen anything, its samples are noise; random
        # commands fill the buffer faster than a planner chasing that noise.
        self.warmup_steps = warmup_steps
        self.steps_taken = 0

        self.current = env.reset()
        self.planner.reset(env.n_agents, env.action_dim)
        self.episode_return = 0.0
        self.episode_coverage = []
        self.finished_episodes = []
        self.believed = []
        self.overlap = []

    def _act(self):
        """One planned decision, or a random one during warm-up."""

        if self.steps_taken < self.warmup_steps:
            actions = np.random.uniform(
                -1.0, 1.0, (self.env.n_agents, self.env.action_dim)
            ).astype(np.float32)
            intent = np.zeros((self.env.n_agents, self.env.n_targets), dtype=np.float32)
            return actions, intent

        beliefs = torch.as_tensor(
            self.current['belief'], dtype=torch.float32, device=self.trajectory.device
        )
        predicted, believed = self.trajectory.predict(beliefs)
        actions, intent = self.planner.plan(
            self.current['camera_states'],
            predicted,
            believed,
            self.current['peer_intent'],
            self.env.action_low,
            self.env.action_high,
            search=self.current.get('search'),
        )
        return self.planner.to_numpy(actions), self.planner.to_numpy(intent)

    def collect(self, num_steps):
        """Step until ``num_steps`` MATE steps are consumed; returns that count."""

        consumed = 0
        while consumed < num_steps:
            actions, intent = self._act()
            nxt, reward, done, info = self.env.step(actions, intent)

            self.buffer.add(
                beliefs=self.current['belief'],
                target_positions=self.current['target_positions'],
                done=done,
            )

            consumed += info['env_steps']
            self.steps_taken += info['env_steps']
            self.episode_return += reward
            self.episode_coverage.append((info['coverage_rate'], info['env_steps']))

            slots = self.current['belief'].reshape(self.env.n_agents, self.env.n_targets, -1)
            self.believed.append(float((slots[..., -1] > 0.5).mean()))
            # How much two cameras' plans claim the same target: 1.0 means no
            # overlap at all, the camera count means they all chase one target.
            self.overlap.append(float(intent.sum(axis=0).max()))

            self.current = nxt

            if done:
                self.finished_episodes.append(
                    {
                        'return': self.episode_return,
                        'coverage_rate': weighted_mean(self.episode_coverage),
                    }
                )
                self.episode_return = 0.0
                self.episode_coverage = []
                self.current = self.env.reset()
                self.planner.reset(self.env.n_agents, self.env.action_dim)

        return consumed

    def drain_stats(self):
        stats = {}
        if self.believed:
            stats['belief/believed_fraction'] = float(np.mean(self.believed))
            stats['plan/max_overlap'] = float(np.mean(self.overlap))
            self.believed, self.overlap = [], []
        if self.finished_episodes:
            coverages = [e['coverage_rate'] for e in self.finished_episodes]
            stats.update(
                {
                    'env/episode_return': float(
                        np.mean([e['return'] for e in self.finished_episodes])
                    ),
                    'env/coverage_rate': float(np.mean(coverages)),
                    'env/coverage_rate_std': float(np.std(coverages)),
                    'env/episodes': len(self.finished_episodes),
                }
            )
            self.finished_episodes = []
        return stats
