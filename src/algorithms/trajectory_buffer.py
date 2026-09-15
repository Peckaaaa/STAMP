"""The only thing the unified pipeline stores: beliefs and where the targets went.

There is no actor, no critic and no state-space world model left in the loop, so
none of what the old replay buffer carried -- observations, global states,
actions, recurrent states -- is read by anything.  What remains is two arrays
per decision, which is roughly a fiftieth of the memory and lets a long run keep
its whole history in RAM.

A sample is a window: the belief every camera held at one decision, and the true
target positions over the next ``horizon`` decisions.  The window never crosses
an episode boundary or the write head, and it comes back **target-major**
(``B, n_targets, horizon, 2``) because that is the layout the trajectory head
predicts in -- transposing at the call site is how the axes got swapped the
first time.
"""

import numpy as np
import torch


class TrajectoryBuffer:
    def __init__(self, capacity, n_agents, n_targets, belief_dim, pin_memory=False):
        self.capacity = capacity
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.size = 0
        self.position = 0
        # Pinned host memory makes the host-to-device copy asynchronous, which
        # matters here because sampling happens every training step.
        self.pin_memory = pin_memory

        self.beliefs = np.zeros((capacity, n_agents, belief_dim), dtype=np.float32)
        self.target_positions = np.zeros((capacity, n_targets, 2), dtype=np.float32)
        self.episode_ids = np.full((capacity,), -1, dtype=np.int64)
        self.current_episode = 0

    def __len__(self):
        return self.size

    def add(self, beliefs, target_positions, done):
        index = self.position
        self.beliefs[index] = beliefs
        self.target_positions[index] = target_positions
        self.episode_ids[index] = self.current_episode

        if done:
            self.current_episode += 1

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------ sampling

    def _to_device(self, array, device):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if self.pin_memory and device.type == 'cuda':
            return tensor.pin_memory().to(device, non_blocking=True)
        return tensor.to(device)

    def sample_windows(self, batch_size, horizon, device):
        """``(beliefs (B, n_agents, belief_dim), future (B, n_targets, horizon, 2))``.

        Returns ``None`` while the buffer holds no window long enough, which is
        what the first few iterations of a run look like.
        """

        device = torch.device(device)
        if self.size < horizon + 2:
            return None

        # Rejection is cheaper than bookkeeping here: candidate starts are drawn
        # in one shot and the invalid ones dropped.
        candidates = np.random.randint(0, self.size - horizon - 1, size=4 * batch_size)
        same_episode = (
            self.episode_ids[candidates] == self.episode_ids[candidates + horizon]
        )
        if self.size == self.capacity:
            # A window that spans the write head mixes the newest and oldest
            # transitions, which are unrelated.
            distance = (self.position - candidates) % self.capacity
            same_episode &= distance > horizon

        starts = candidates[same_episode][:batch_size]
        if starts.size == 0:
            return None

        offsets = starts[:, None] + np.arange(1, horizon + 1)[None, :]
        beliefs = self._to_device(self.beliefs[starts], device)
        # (B, horizon, n_targets, 2) -> target-major, the head's own layout.
        future = self._to_device(
            self.target_positions[offsets].transpose(0, 2, 1, 3), device
        )
        return beliefs, future
