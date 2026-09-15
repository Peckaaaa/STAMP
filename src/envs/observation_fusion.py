"""Target-slot extraction, peer-to-peer merging, and the oracle fusion ceiling.

MATE lays every camera's observation out with one fixed slot per target, carrying
that target's public state and a bit saying whether the camera can see it right
now.  Slot ``j`` is the same target for every camera, so merging two cameras'
views is a per-slot choice and needs no data association -- which is what makes
the peer-to-peer belief in :mod:`envs.mate_wrapper` cheap and exact.

Two consumers live here:

``merge_slots``              the deployable one -- each camera merges what it
                            sees with what its neighbours told it.
``fuse_camera_observations`` the oracle -- every camera receives the union of
                            the whole team's field of view, instantly and for
                            free.  Not deployable; it is the ceiling the learned
                            peer-to-peer belief is measured against, and the
                            Phase 0 sweep reports the gap.

MATE ships the oracle as ``mate.wrappers.SharedFieldOfView``, but the wrapper
cannot be used here: ``SingleTeamHelper.__init__`` in this checkout walks down
``env.env`` until it reaches the base ``MultiAgentTracking`` and wraps *that*, so
``MultiCamera(SharedFieldOfView(base))`` silently drops the fusion.

A camera observes the *exact* public state of any entity inside its field of
view, so merging two cameras' views of one slot is a choice between identical
vectors, not an average.
"""

import numpy as np

from envs.config_resolver import ensure_mate_importable


def _constants():
    ensure_mate_importable()
    from mate import constants as consts

    return consts


def camera_slices(num_cameras, num_targets, num_obstacles):
    return _constants().camera_observation_slices_of(num_cameras, num_targets, num_obstacles)


def target_slot_dim():
    """Public target state plus the visibility bit that follows it."""

    return _constants().TARGET_STATE_DIM_PUBLIC + 1


def camera_target_slots(joint_observation, num_cameras, num_targets, num_obstacles):
    """``(num_cameras, obs_dim)`` -> ``(num_cameras, num_targets, slot_dim)``.

    This block is what a camera knows about the targets and, in the peer-to-peer
    channel, exactly what it has to say to its neighbours.
    """

    slices = camera_slices(num_cameras, num_targets, num_obstacles)
    observation = np.asarray(joint_observation, dtype=np.float64).reshape(num_cameras, -1)
    return observation[:, slices['opponent_states_with_mask']].reshape(
        num_cameras, num_targets, target_slot_dim()
    )


def merge_slots(own, received):
    """Merge one camera's slots with the slots its neighbours sent.

    ``own`` is ``(num_targets, slot_dim)``; ``received`` is
    ``(num_senders, num_targets, slot_dim)``.  Returns the merged block and, per
    slot, whether the knowledge came from the camera itself.

    What the camera sees wins: its own observation is current, while a peer's is
    at best one exchange old.  Among peers the first sender with the slot wins,
    and since a visible slot holds the exact public state they all agree anyway.
    """

    merged = np.array(own, dtype=np.float64, copy=True)
    first_hand = merged[:, -1] > 0.0

    if len(received) == 0:
        return merged, first_hand

    received = np.asarray(received, dtype=np.float64)
    sender_has = received[..., -1] > 0.0            # (senders, targets)
    peer_has = sender_has.any(axis=0)               # (targets,)
    source = np.argmax(sender_has, axis=0)          # first sender holding each slot

    take = peer_has & ~first_hand
    merged[take] = received[source[take], np.flatnonzero(take)]
    return merged, first_hand


def _fuse_slots(observation, entity_slice, state_dim, num_slots):
    """OR the per-slot masks across cameras and broadcast the observed states.

    ``observation`` is modified in place.  Every slot block is
    ``[state (state_dim), mask (1)]`` repeated ``num_slots`` times, and slot
    ``j`` is the same entity for every camera.
    """

    num_cameras = observation.shape[0]
    block = observation[:, entity_slice].reshape(num_cameras, num_slots, state_dim + 1)

    seen_by = block[..., -1] > 0.0                      # (num_cameras, num_slots)
    seen = seen_by.any(axis=0)                          # (num_slots,)
    source = np.argmax(seen_by, axis=0)                 # first camera that saw each slot

    fused = block[source, np.arange(num_slots)].copy()  # (num_slots, state_dim + 1)
    fused[~seen] = 0.0

    block[:] = fused[np.newaxis]
    observation[:, entity_slice] = block.reshape(num_cameras, -1)


def fuse_camera_observations(joint_observation, num_cameras, num_targets, num_obstacles):
    """``(num_cameras, obs_dim)`` -> the same observations with a shared field of view.

    Targets, obstacles and teammates are fused; the preserved data and each
    camera's own private state are left alone.
    """

    consts = _constants()
    slices = camera_slices(num_cameras, num_targets, num_obstacles)
    fused = np.array(joint_observation, dtype=np.float64, copy=True).reshape(num_cameras, -1)

    _fuse_slots(
        fused, slices['opponent_states_with_mask'], consts.TARGET_STATE_DIM_PUBLIC, num_targets
    )
    _fuse_slots(fused, slices['obstacle_states_with_mask'], consts.OBSTACLE_STATE_DIM, num_obstacles)
    _fuse_slots(
        fused, slices['teammate_states_with_mask'], consts.CAMERA_STATE_DIM_PUBLIC, num_cameras
    )
    return fused


def target_visibility(joint_observation, num_cameras, num_targets, num_obstacles):
    """Per-camera target masks ``(num_cameras, num_targets)`` as booleans."""

    slices = camera_slices(num_cameras, num_targets, num_obstacles)
    observation = np.asarray(joint_observation, dtype=np.float64).reshape(num_cameras, -1)
    return observation[:, slices['opponent_mask']] > 0.0


def camera_positions_from_state(state, num_cameras):
    """``(num_cameras, 2)`` camera locations out of MATE's global state vector.

    The state concatenates preserved data, then every camera's private state,
    then the targets', so a camera's location is the first pair of its block.
    """

    consts = _constants()
    state = np.asarray(state, dtype=np.float64)
    offset = consts.PRESERVED_DIM
    stride = consts.CAMERA_STATE_DIM_PRIVATE
    return np.stack([state[offset + i * stride : offset + i * stride + 2] for i in range(num_cameras)])


def target_positions_from_state(state, num_cameras, num_targets):
    """``(num_targets, 2)`` true target locations, for supervising predictions.

    Global state, so this is training-time information only -- it never reaches
    a camera's own belief, which is what makes it a legitimate CTDE label.
    """

    consts = _constants()
    state = np.asarray(state, dtype=np.float64)
    offset = consts.PRESERVED_DIM + num_cameras * consts.CAMERA_STATE_DIM_PRIVATE
    stride = consts.TARGET_STATE_DIM_PRIVATE
    return np.stack([state[offset + j * stride : offset + j * stride + 2] for j in range(num_targets)])
