"""MATE wrapper that runs the peer-to-peer channel and builds each camera's belief.

One ``step`` yields ``(s_t, o_t^{1:n}, b_t^{1:n}, a_t^{1:n}, r_t, s_{t+1}, done)``
with continuous camera actions.  Only the camera team is controlled; the targets
run MATE's ``GreedyTargetAgent``.

What travels on the channel is the sender's target-slot block -- the public state
of every target it can currently see -- addressed to each neighbour in turn, not
broadcast to the team.  That is the protocol MATE's own ``GreedyCameraAgent``
implements by hand, and it is what makes a receiver able to merge: slot ``j`` is
the same target for every camera, so a merge is a per-slot choice with no data
association.  A learned message vector could carry the same information, but
nothing downstream could then be held to a shared meaning -- the belief a camera
publishes and the trajectory it predicts have to be comparable across cameras
for the consensus term to say anything.

The belief each camera ends a step with is that merge plus two columns the
observation cannot provide: how many decisions have passed since anything
refreshed the slot, and whether the camera saw it itself or was told.  Age is
what lets a policy tell "seen just now" from "remembered from a while ago"; the
first-hand bit is what keeps two cameras' beliefs distinguishable at all, since
a fully connected team merges to the same union and would otherwise hold
identical beliefs -- measured, and it drove the consensus term to exactly zero.

``MATE-main`` is a gymnasium port of MATE, so ``reset`` returns
``(observation, info)`` and ``step`` returns the five-tuple.
"""

import numpy as np
import torch

from envs.config_resolver import ensure_mate_importable, resolve_scenario
from envs.observation_fusion import (
    camera_positions_from_state,
    camera_target_slots,
    fuse_camera_observations,
    merge_slots,
    target_positions_from_state,
    target_slot_dim,
)


class RunningMeanStd:
    """Welford statistics for observation and state normalization.

    MATE's observation and state spaces both carry ``+inf`` upper bounds (target
    bounties and warehouse counters are unbounded), so min-max rescaling against
    the declared Box is not usable; empirical statistics are.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        out = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -clip, clip).astype(np.float32)

    def denormalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
            return x * std + mean
        return np.asarray(x) * np.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class MATEEnv:
    """MATE ``MultiAgentTracking`` with continuous camera control and a P2P channel.

    Actions handed to :meth:`step` are normalized to ``[-1, 1]`` per dimension and
    rescaled to MATE's camera Box (``[-5, 5] x [-2.5, 2.5]``) here.
    """

    #: Beyond this many decisions a stale slot is simply "old"; the counter
    #: saturates so the policy sees a bounded input.
    MAX_AGE = 25.0

    #: Cells per axis in the team's staleness map.  The terrain is 2000 units
    #: wide, so a cell is 125 units -- about a tenth of a camera's sight range,
    #: fine enough that a sector covers a handful of cells and coarse enough
    #: that the whole grid is one small tensor per decision.
    GRID_SIZE = 16
    #: A cell nobody has looked at for this many decisions is simply
    #: "unvisited"; saturating keeps one forgotten corner from outweighing the
    #: rest of the map for the whole episode.
    MAX_STALENESS = 40.0

    def __init__(
        self,
        scenario='MATE-4v8-9',
        max_episode_steps=200,
        camera_comm=True,
        comm_range=None,
        reward_scale=1.0,
        frame_skip=1,
        seed=0,
        normalize=True,
        shared_fov=False,
        update_statistics=True,
    ):
        ensure_mate_importable()

        import mate
        from mate.agents import GreedyTargetAgent

        self.scenario = resolve_scenario(scenario)
        self.camera_comm = camera_comm
        # None means every teammate is a neighbour.  A finite range is measured
        # between camera locations, which are fixed physical facts about the
        # deployment rather than anything a camera has to perceive.
        self.comm_range = comm_range
        self.reward_scale = reward_scale
        self.max_episode_steps = max_episode_steps
        self.frame_skip = int(frame_skip)
        # Oracle fusion: every camera observes the union of the team's field of
        # view.  Not a deployable setting -- it is the ceiling the peer-to-peer
        # belief is measured against.
        self.shared_fov = bool(shared_fov)

        # make_environment() instead of gym.make(): it builds MultiAgentTracking
        # directly, with no checker wrapper in front of MATE's
        # (camera_obs, target_obs) joint-observation tuple.
        base_env = mate.make_environment(
            config=self.scenario, max_episode_steps=max_episode_steps
        )
        self.env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())

        unwrapped = self.env.unwrapped
        self.n_agents = unwrapped.num_cameras
        self.n_targets = unwrapped.num_targets
        self.n_obstacles = unwrapped.num_obstacles
        self.obs_dim = unwrapped.camera_observation_space.shape[0]
        self.state_dim = unwrapped.state_space.shape[0]
        self.action_dim = unwrapped.camera_action_space.shape[0]

        self.slot_dim = target_slot_dim()
        #: What one camera puts on the wire: its target-slot block, and what its
        #: current plan intends to cover.  The intent is what keeps four cameras
        #: that share a belief from planning the same sweep -- coverage counts
        #: distinct targets, and nothing in an independent score says so.
        self.msg_dim = self.n_targets * (self.slot_dim + 1)
        #: What it reads afterwards: the merged state, whether it is first
        #: hand, how stale it is, and whether the slot is known at all.
        self.belief_dim = self.n_targets * (self.slot_dim + 2)

        self.action_low = unwrapped.camera_action_space.low.astype(np.float64)
        self.action_high = unwrapped.camera_action_space.high.astype(np.float64)

        self.normalize = normalize
        # Evaluation should read the distribution the policy was trained under.
        # Folding evaluation rollouts back into the running statistics moves the
        # scale of every observation the actor sees while it is being measured.
        self.update_statistics = update_statistics
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.state_rms = RunningMeanStd((self.state_dim,))

        self.age = np.zeros((self.n_agents, self.n_targets), dtype=np.float64)
        self.peer_intent = np.zeros((self.n_agents, self.n_targets), dtype=np.float64)
        self.episode_step = 0

        # Spatial memory, kept beside the belief rather than inside it: the
        # belief vector is what the trajectory head is conditioned on, and every
        # checkpoint is tied to its width, so a new state that only the planner
        # reads must not widen it.
        self.cell_centres = self._grid_centres()
        self.staleness = np.zeros(
            (self.n_agents, self.cell_centres.shape[0]), dtype=np.float64
        )
        # Where a slot was last actually fixed, in map units.  A stale slot's
        # coordinates are zeroed out of the belief once nobody sees it, so the
        # last fix has nowhere else to live.
        self.last_seen = np.zeros((self.n_agents, self.n_targets, 2), dtype=np.float64)
        # MATE-main's wrapper.seed() still calls the RandomState-only randint on
        # gymnasium's Generator, so seeding goes through the first reset instead.
        self._pending_seed = seed

    # ------------------------------------------------------------------ helpers

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def describe(self):
        return (
            f'{self.scenario}: {self.n_agents} cameras, {self.n_targets} targets, '
            f'{self.n_obstacles} obstacles | state {self.state_dim}, obs {self.obs_dim}, '
            f'action {self.action_dim}, P2P payload {self.msg_dim}, belief {self.belief_dim}, '
            f'frame skip {self.frame_skip}'
            + (f', comm range {self.comm_range:.0f}' if self.comm_range else '')
            + (' | shared field of view' if self.shared_fov else '')
            + ('' if self.camera_comm else ' | channel off')
        )

    def _norm_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(self.n_agents, self.obs_dim)
        if self.shared_fov:
            # Before normalization: the running statistics have to describe the
            # observations the policy is actually handed.
            obs = fuse_camera_observations(
                obs, self.n_agents, self.n_targets, self.n_obstacles
            )
        if not self.normalize:
            return obs.astype(np.float32)
        if self.update_statistics:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _norm_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.state_dim)
        if not self.normalize:
            return state.astype(np.float32)
        if self.update_statistics:
            self.state_rms.update(state[None])
        return self.state_rms.normalize(state)

    def _scale_action(self, actions):
        """``[-1, 1]`` per dimension -> MATE's camera Box."""

        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        actions = actions.reshape(self.n_agents, self.action_dim)
        return self.action_low + 0.5 * (actions + 1.0) * (self.action_high - self.action_low)

    def _neighbours(self):
        """``(n_agents, n_agents)`` boolean: who is close enough to talk to whom."""

        connected = ~np.eye(self.n_agents, dtype=bool)
        if not self.comm_range:
            return connected

        positions = camera_positions_from_state(
            self.env.unwrapped.state(), self.n_agents
        )
        distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        return connected & (distance <= self.comm_range)

    def camera_states(self):
        """``(n_agents, 9)`` raw private camera states, unnormalized.

        Position, sight vector, viewing angle and the three constants a planner
        needs to roll the optics forward.  Every entry is the camera's own,
        so this is not privileged information.
        """

        ensure_mate_importable()
        from mate import constants as consts

        state = self.env.unwrapped.state()
        offset = consts.PRESERVED_DIM
        stride = consts.CAMERA_STATE_DIM_PRIVATE
        return np.stack(
            [state[offset + i * stride : offset + (i + 1) * stride] for i in range(self.n_agents)]
        )

    def _grid_centres(self):
        """``(GRID_SIZE ** 2, 2)`` cell centres in map units."""

        ensure_mate_importable()
        from mate import constants as consts

        edge = 2.0 * consts.TERRAIN_SIZE / self.GRID_SIZE
        axis = -consts.TERRAIN_SIZE + edge * (np.arange(self.GRID_SIZE) + 0.5)
        x, y = np.meshgrid(axis, axis, indexing='xy')
        return np.stack([x.ravel(), y.ravel()], axis=-1)

    def _covered_cells(self, camera_states):
        """``(n_agents, n_cells)`` boolean: which cells each sector contains.

        The same range-and-bearing test MATE uses for detection, minus the
        obstacles: a cell is a place, not a target, and nothing is hiding behind
        anything.  Hard, not soft -- this decides what "has been looked at"
        means, and a soft answer would leave every cell partly stale forever.
        """

        location = camera_states[:, 0:2]
        sight = camera_states[:, 3:5]
        sight_range = np.linalg.norm(sight, axis=-1)
        orientation = np.degrees(np.arctan2(sight[:, 1], sight[:, 0]))
        viewing_angle = camera_states[:, 5]

        relative = self.cell_centres[None, :, :] - location[:, None, :]
        distance = np.linalg.norm(relative, axis=-1)
        bearing = np.degrees(np.arctan2(relative[..., 1], relative[..., 0]))
        offset = np.abs((bearing - orientation[:, None] + 180.0) % 360.0 - 180.0)
        return (distance <= sight_range[:, None]) & (offset <= 0.5 * viewing_angle[:, None])

    def _update_spatial_memory(self, beliefs, camera_states):
        """Age the staleness map and record where each slot was last fixed.

        Staleness is per camera, not global, and that is a CTDE requirement
        rather than a refinement.  The map is read at *execution* time, inside
        the planner, at every decision -- so anything centralized in it would be
        centralized execution.  A camera may therefore only zero the cells its
        own sector covers and the cells its neighbours covered, because a
        neighbour's sector is exactly what the peer-to-peer round already tells
        it.  Out of range, two cameras disagree about what the team has seen;
        that disagreement is the decentralized state, and it collapses to one
        shared map when everyone is connected.

        The centralized side of CTDE stays where it belongs: the true target
        positions the trajectory head regresses onto, used in training only.
        """

        covered = self._covered_cells(camera_states)
        # A camera hears itself as well as whoever is close enough to talk.
        reach = self._neighbours() | np.eye(self.n_agents, dtype=bool)
        heard = reach.astype(np.float64) @ covered.astype(np.float64) > 0.0

        self.staleness = np.where(
            heard, 0.0, np.minimum(self.staleness + 1.0, self.MAX_STALENESS)
        )

        known = beliefs[..., -1] > 0.0
        self.last_seen = np.where(known[..., None], beliefs[..., 0:2], self.last_seen)

    def _search_state(self):
        """What the planner needs that the belief cannot carry.

        Everything is in map units, the same coordinates the planner rolls the
        optics in, so nothing downstream has to know about belief scaling.
        """

        return {
            'cell_centres': self.cell_centres.astype(np.float32),
            'staleness': (self.staleness / self.MAX_STALENESS).astype(np.float32),
            'last_seen': self.last_seen.astype(np.float32),
            'age': self.age.astype(np.float32),
            'known': (self.age == 0.0).astype(np.float32),
        }

    def _exchange_slots(self, raw_observation, intents):
        """Run one peer-to-peer round and return what each camera received.

        Every camera addresses its neighbours individually -- ``recipient=j``,
        not a broadcast -- so routing, and therefore any communication-range,
        delay or dropout wrapper MATE has applied, is done by the environment,
        and the receiver knows which camera each block came from.
        """

        own = camera_target_slots(
            raw_observation, self.n_agents, self.n_targets, self.n_obstacles
        )
        received = [[] for _ in range(self.n_agents)]
        peer_intent = np.zeros((self.n_agents, self.n_targets))
        if not self.camera_comm:
            return own, received, peer_intent

        from mate.utils import Message, Team

        connected = self._neighbours()
        outgoing = []
        for sender in range(self.n_agents):
            payload = np.concatenate(
                [own[sender].astype(np.float64).ravel(), intents[sender].astype(np.float64)]
            )
            for recipient in np.flatnonzero(connected[sender]):
                outgoing.append(
                    Message(
                        sender=sender,
                        recipient=int(recipient),
                        content=payload.copy(),
                        team=Team.CAMERA,
                    )
                )
        self.env.send_messages(outgoing)

        slot_payload = self.n_targets * self.slot_dim
        for index, inbox in enumerate(self.env.receive_messages()):
            for message in inbox:
                if message.sender == index:
                    continue
                content = np.asarray(message.content, dtype=np.float64)
                received[index].append(
                    content[:slot_payload].reshape(self.n_targets, self.slot_dim)
                )
                # A neighbour's claim is a lower bound on how covered a target
                # already is; the strongest claim is the one worth avoiding.
                peer_intent[index] = np.maximum(peer_intent[index], content[slot_payload:])
        return own, received, peer_intent

    def _belief(self, raw_observation, intents, reset=False):
        """Merge the channel into a per-camera belief and age every slot."""

        own, received, peer_intent = self._exchange_slots(raw_observation, intents)
        self.peer_intent = peer_intent

        if reset:
            self.age[:] = 0.0
            # A new episode is a new map: nothing has been looked at yet, so
            # every cell starts maximally stale rather than fresh, or the first
            # decisions of an episode would see no reason to search at all.
            self.staleness[:] = self.MAX_STALENESS
            self.last_seen[:] = 0.0

        # Slot layout: [x, y, sight range, loaded, first hand, age, known].
        # The known bit stays last so every consumer can find it the same way.
        beliefs = np.zeros((self.n_agents, self.n_targets, self.slot_dim + 2))
        for index in range(self.n_agents):
            merged, first_hand = merge_slots(own[index], received[index])
            known = merged[:, -1] > 0.0

            self.age[index] = np.where(
                known, 0.0, np.minimum(self.age[index] + 1.0, self.MAX_AGE)
            )
            beliefs[index, :, : self.slot_dim - 1] = merged[:, :-1]
            beliefs[index, :, -3] = first_hand.astype(np.float64)
            beliefs[index, :, -2] = self.age[index] / self.MAX_AGE
            beliefs[index, :, -1] = merged[:, -1]

        # Coordinates are still in map units here, which is what the spatial
        # memory and the planner both work in.
        self._update_spatial_memory(beliefs, self.camera_states())

        return self._scale_belief(beliefs)

    def _zero_intents(self):
        return np.zeros((self.n_agents, self.n_targets), dtype=np.float32)

    def _scale_belief(self, beliefs):
        """Fixed scaling, not running statistics.

        A slot's mask and age are bounded by construction and its coordinates by
        the terrain, so the scale is known in advance; running statistics would
        drift with the policy and blur the difference between an empty slot and
        a remembered one.
        """

        ensure_mate_importable()
        from mate import constants as consts

        scaled = np.array(beliefs, dtype=np.float32)
        scaled[..., 0:2] /= consts.TERRAIN_SIZE          # location
        scaled[..., 2] /= consts.TERRAIN_SIZE            # sight range
        return scaled.reshape(self.n_agents, self.belief_dim)

    def _target_positions(self):
        """True target locations in belief coordinates.

        Global state, so this is a training-time label for the trajectory head
        and never part of what a camera knows.
        """

        ensure_mate_importable()
        from mate import constants as consts

        positions = target_positions_from_state(
            self.env.unwrapped.state(), self.n_agents, self.n_targets
        )
        return (positions / consts.TERRAIN_SIZE).astype(np.float32)

    # ------------------------------------------------------------------- env API

    def reset(self):
        self.episode_step = 0
        if self._pending_seed is not None:
            obs, _ = self.env.reset(seed=self._pending_seed)
            self._pending_seed = None
        else:
            obs, _ = self.env.reset()

        raw = np.asarray(obs, dtype=np.float64).reshape(self.n_agents, self.obs_dim)
        if self.shared_fov:
            raw = fuse_camera_observations(raw, self.n_agents, self.n_targets, self.n_obstacles)

        belief = self._belief(raw, self._zero_intents(), reset=True)
        return {
            'state': self._norm_state(self.env.unwrapped.state()),
            'obs': self._norm_obs(obs),
            'belief': belief,
            'peer_intent': self.peer_intent.astype(np.float32),
            'camera_states': self.camera_states(),
            'target_positions': self._target_positions(),
            'search': self._search_state(),
        }

    def step(self, actions, intents=None):
        """Hold the action for ``frame_skip`` MATE steps, then rebuild the belief.

        The peer-to-peer round runs once per decision, not once per MATE step:
        the policy only gets to speak when it gets to act.  Every returned rate
        is the mean over the MATE steps this decision actually consumed, so an
        episode mean over decisions still equals the per-MATE-step mean when
        weighted by ``info['env_steps']`` -- which is what MATE's coverage-rate
        metric is defined over.

        Args:
            actions: ``(n_agents, action_dim)`` in ``[-1, 1]``.
            intents: ``(n_agents, n_targets)`` what each camera's plan means to
                cover, published to its neighbours.  ``None`` publishes nothing,
                which is what a policy without a plan has to say.

        Returns:
            ``(next, reward, done, info)``.
        """

        if intents is None:
            intents = self._zero_intents()
        scaled = self._scale_action(actions)

        rates = {'coverage_rate': 0.0, 'real_coverage_rate': 0.0, 'mean_transport_rate': 0.0}
        consumed = 0
        done = False
        for _ in range(self.frame_skip):
            obs, _, terminated, truncated, infos = self.env.step(scaled)
            # MATE-main folds the episode-step limit into `terminated`; `truncated`
            # is always False, but both are honored here.
            done = bool(terminated) or bool(truncated)
            self.episode_step += 1
            consumed += 1

            # MATE's raw team reward is unbounded below (measured min -198); its
            # own coverage_rate metric is the team-mean tracking rate in [0, 1]
            # and is what this task optimizes.
            for key in rates:
                rates[key] += float(infos[0][key])

            if done:
                break

        for key in rates:
            rates[key] /= consumed

        raw = np.asarray(obs, dtype=np.float64).reshape(self.n_agents, self.obs_dim)
        if self.shared_fov:
            raw = fuse_camera_observations(raw, self.n_agents, self.n_targets, self.n_obstacles)

        belief = self._belief(raw, intents)
        nxt = {
            'state': self._norm_state(self.env.unwrapped.state()),
            'obs': self._norm_obs(obs),
            'belief': belief,
            'peer_intent': self.peer_intent.astype(np.float32),
            'camera_states': self.camera_states(),
            'target_positions': self._target_positions(),
            'search': self._search_state(),
        }
        info = dict(rates, env_steps=consumed)
        return nxt, rates['coverage_rate'] * self.reward_scale, done, info
