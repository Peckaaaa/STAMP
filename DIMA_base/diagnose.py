"""Where the 0.26 coverage gap to the greedy rule agent actually lives.

Phase 0 measured the gap; this measures its cause.  The world-model pipeline
hands the policy a chain of three lossy stages, and a failure in any one of them
caps coverage no matter how good the others are:

    s_t -> FSQ tokens -> decoded ô_t -> actor -> action

Two diagnostics cut that chain at its two joints, and neither needs a trained
world model -- which matters, because a diffusion run does not fit on a CPU box:

  tokenizer   Can 16 FSQ tokens (144 bits) even carry the target positions the
              policy has to act on?  Trains the tokenizer alone on greedy
              rollouts and reports the reconstruction error of the target slots,
              in map units, against a larger token budget as a control.  If the
              decoded observation loses the targets, imagination quality and PPO
              tuning are both irrelevant.

  clone       Can this actor, on this observation, reach greedy coverage at all?
              Fits the actor network to GreedyCameraAgent's own actions by
              supervised regression -- no RL, no world model, no credit
              assignment -- and evaluates the result in the real environment.
              Behaviour cloning is an upper bound on what the policy class can
              extract from a single-step observation: if it lands near 0.60 the
              architecture is sufficient and the loss is downstream, and if it
              lands near 0.40 the observation the actor is handed is the limit.

Both share one greedy-driven dataset, cached in the scratchpad.

Run:
    python diagnose.py dataset --episodes 50
    python diagnose.py tokenizer
    python diagnose.py clone
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from envs.config_resolver import ensure_mate_importable, resolve_scenario
from envs.mate_wrapper import MATEEnv, RunningMeanStd
from envs.observation_fusion import fuse_camera_observations


DEFAULT_CACHE = os.path.join('runs', 'diagnose', 'greedy_dataset.npz')
DEFAULT_CONFIG = os.path.join('src', 'configs', 'default.yaml')


# --------------------------------------------------------------------- dataset


def collect_greedy_dataset(scenario, episodes, max_episode_steps, base_seed):
    """Greedy camera agents against greedy targets, recording raw observations.

    Raw, not normalized: the rule agent reads MATE's own observation vector, and
    the statistics used to normalize it are fitted afterwards on exactly this
    data, so the cloned actor and the tokenizer both see what the environment
    wrapper would have handed them.
    """

    import mate
    from gymnasium.utils import seeding
    from mate.agents import GreedyCameraAgent, GreedyTargetAgent

    base_env = mate.make_environment(config=scenario, max_episode_steps=max_episode_steps)
    env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())
    unwrapped = env.unwrapped

    low = unwrapped.camera_action_space.low.astype(np.float64)
    high = unwrapped.camera_action_space.high.astype(np.float64)

    counts = (unwrapped.num_cameras, unwrapped.num_targets, unwrapped.num_obstacles)
    observations, fused_observations, actions, states, coverages = [], [], [], [], []

    for episode in range(episodes):
        prototype = GreedyCameraAgent()
        prototype.seed(base_seed + episode)
        agents = prototype.spawn(unwrapped.num_cameras)

        for offset, target_agent in enumerate(env.opponent_agents_ordered):
            target_agent._np_random, _ = seeding.np_random(
                base_seed + episode + 1000 * (offset + 1)
            )
        observation, _ = env.reset(seed=base_seed + episode)
        mate.group_reset(agents, observation)
        infos = None

        coverage = []
        done = False
        while not done:
            joint_action = mate.group_step(unwrapped, agents, observation, infos)
            joint_action = np.asarray(joint_action, dtype=np.float64)

            observations.append(np.asarray(observation, dtype=np.float32))
            # The same step under oracle fusion.  The demonstrator still acts on
            # its own partial view -- only the features stored beside its action
            # change -- so cloning from this column asks whether the team's
            # union of sight is what the actor was missing.
            fused_observations.append(
                fuse_camera_observations(observation, *counts).astype(np.float32)
            )
            # Back to the wrapper's [-1, 1] box, which is what the actor emits.
            actions.append(
                (2.0 * (joint_action - low) / (high - low) - 1.0).astype(np.float32)
            )
            states.append(np.asarray(unwrapped.state(), dtype=np.float32))

            observation, _, terminated, truncated, infos = env.step(joint_action)
            done = bool(terminated) or bool(truncated)
            coverage.append(float(infos[0]['coverage_rate']))

        coverages.append(float(np.mean(coverage)))

    env.close()
    return {
        'obs': np.stack(observations),
        'obs_fused': np.stack(fused_observations),
        'actions': np.stack(actions),
        'states': np.stack(states),
        'coverage': np.asarray(coverages, dtype=np.float32),
        'num_cameras': unwrapped.num_cameras,
        'num_targets': unwrapped.num_targets,
        'num_obstacles': unwrapped.num_obstacles,
    }


def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No dataset at {path}. Run `python diagnose.py dataset` first.')
    data = np.load(path)
    return {key: data[key] for key in data.files}


def fitted_statistics(dataset, key='obs'):
    """Running statistics over the dataset, matching what ``MATEEnv`` would build."""

    obs_rms = RunningMeanStd((dataset[key].shape[-1],))
    obs_rms.update(dataset[key].reshape(-1, dataset[key].shape[-1]))

    state_rms = RunningMeanStd((dataset['states'].shape[-1],))
    state_rms.update(dataset['states'])
    return obs_rms, state_rms


def previous_actions(dataset, episode_length):
    """Each camera's own action one decision earlier, zeros at an episode start.

    ``GreedyCameraAgent`` repeats ``prev_action`` whenever it currently sees no
    target, and at this scenario's visibility -- under a fifth of the target
    slots -- that branch drives much of its behaviour.  A memoryless actor
    cannot represent it, so this column measures how much of the imitation gap
    is the missing memory of what the camera was already doing.
    """

    actions = dataset['actions']
    episodes = actions.shape[0] // episode_length
    assert episodes * episode_length == actions.shape[0], (
        f'{actions.shape[0]} decisions do not divide into episodes of {episode_length}'
    )

    shaped = actions.reshape(episodes, episode_length, *actions.shape[1:])
    previous = np.zeros_like(shaped)
    previous[:, 1:] = shaped[:, :-1]
    return previous.reshape(actions.shape)


def stack_history(observations, length, episode_length):
    """Concatenate the last ``length`` observations at each step, newest first.

    Steps near an episode start repeat their earliest observation, which is
    exactly what the driving rollout does before its window has filled, so the
    offline fit and the online policy read the same feature layout.

    ``GreedyCameraAgent`` decides from a 25-step memory of where targets were,
    and at this scenario's visibility most of its commands are issued while
    nothing is in view.  A window is the cheapest way to ask whether that memory
    is the variable a single-step actor is missing.
    """

    if length <= 1:
        return observations

    episodes = observations.shape[0] // episode_length
    assert episodes * episode_length == observations.shape[0], (
        f'{observations.shape[0]} decisions do not divide into episodes of {episode_length}'
    )

    shaped = observations.reshape(episodes, episode_length, *observations.shape[1:])
    lags = [shaped[:, np.clip(np.arange(episode_length) - lag, 0, None)] for lag in range(length)]
    stacked = np.concatenate(lags, axis=-1)
    return stacked.reshape(observations.shape[0], observations.shape[1], -1)


def push_history(window, observation, length):
    """Newest-first window of ``length`` observations, filled by repetition."""

    if not window:
        window.extend([observation] * length)
    else:
        window.insert(0, observation)
        del window[length:]
    return np.concatenate(window, axis=-1)


def split(count, holdout=0.2, seed=0):
    order = np.random.default_rng(seed).permutation(count)
    cut = int(count * (1.0 - holdout))
    return order[:cut], order[cut:]


# ------------------------------------------------------------------- tokenizer


def target_slot_error(decoded, truth, dataset, obs_rms):
    """Position error of the observed targets, in map units, on visible slots only.

    Only slots a camera actually sees carry a position; the rest are zeros the
    decoder can trivially match, and averaging them in would hide the failure.
    """

    ensure_mate_importable()
    from mate import constants as consts

    counts = (
        int(dataset['num_cameras']),
        int(dataset['num_targets']),
        int(dataset['num_obstacles']),
    )
    slices = consts.camera_observation_slices_of(*counts)
    dim = consts.TARGET_STATE_DIM_PUBLIC

    decoded = obs_rms.denormalize(decoded.detach().cpu().numpy())
    truth = obs_rms.denormalize(truth.detach().cpu().numpy())

    block = lambda x: x[..., slices['opponent_states_with_mask']].reshape(
        *x.shape[:-1], counts[1], dim + 1
    )
    predicted_block, true_block = block(decoded), block(truth)

    visible = true_block[..., -1] > 0.5
    if not visible.any():
        return float('nan'), float('nan'), 0.0

    position_error = np.linalg.norm(
        predicted_block[..., :2] - true_block[..., :2], axis=-1
    )[visible]
    # The mask bit is what tells the policy a slot means anything at all.
    mask_error = np.abs(predicted_block[..., -1] - true_block[..., -1]).mean()
    return float(position_error.mean()), float(mask_error), float(visible.mean())


def run_tokenizer(args):
    from models.state_autoencoder import StateAutoEncoder

    dataset = load_dataset(args.cache)
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    obs_rms, state_rms = fitted_statistics(dataset)

    device = torch.device(args.device)
    n_agents = int(dataset['num_cameras'])

    obs = torch.as_tensor(obs_rms.normalize(dataset['obs']), device=device)
    states = torch.as_tensor(state_rms.normalize(dataset['states']), device=device)
    messages = torch.zeros(states.shape[0], n_agents, config['env']['msg_dim'], device=device)

    train_index, test_index = split(states.shape[0], seed=args.seed)
    train_index = torch.as_tensor(train_index, device=device)
    test_index = torch.as_tensor(test_index, device=device)

    print(f'{states.shape[0]} transitions, {len(train_index)} train / {len(test_index)} held out')
    print(
        f'target visibility in the data: '
        f'{float((dataset["obs"] != 0).mean()):.3f} nonzero observation entries'
    )

    for num_tokens in args.tokens:
        torch.manual_seed(args.seed)
        autoencoder = StateAutoEncoder(
            state_dim=states.shape[-1],
            obs_dim=obs.shape[-1],
            msg_dim=config['env']['msg_dim'],
            n_agents=n_agents,
            levels=tuple(config['world_model']['levels']),
            num_tokens=num_tokens,
            hidden_dim=config['world_model']['ae_hidden'],
        ).to(device)
        optimizer = torch.optim.Adam(
            autoencoder.parameters(), lr=config['world_model']['lr']
        )

        for step in range(args.updates):
            batch = train_index[torch.randint(len(train_index), (args.batch_size,), device=device)]
            loss, _ = autoencoder.loss(states[batch], obs[batch], messages[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), config['world_model']['grad_clip']
            )
            optimizer.step()

        with torch.no_grad():
            hat_state, hat_obs, _, indices = autoencoder(states[test_index], messages[test_index])
            state_mse = F.mse_loss(hat_state, states[test_index]).item()
            obs_mse = F.mse_loss(hat_obs, obs[test_index]).item()
            usage = indices.unique().numel() / float(autoencoder.codebook_size)

        position_error, mask_error, visible_fraction = target_slot_error(
            hat_obs, obs[test_index], dataset, obs_rms
        )
        bits = num_tokens * np.log2(autoencoder.codebook_size)
        print(
            f'  tokens {num_tokens:>3} ({bits:.0f} bits): held-out state MSE {state_mse:.4f}, '
            f'obs MSE {obs_mse:.4f} (1.0 = predicting the mean) | codebook usage {usage:.3f}\n'
            f'  {"":>16} visible target position error {position_error:8.1f} map units '
            f'(terrain is {2000} wide), mask error {mask_error:.3f}, '
            f'{visible_fraction:.3f} of slots visible'
        )


# ------------------------------------------------------------- behaviour clone


def run_clone(args):
    from algorithms.communicative_mappo import CommunicativeMAPPO
    from evaluate import evaluate_policy

    dataset = load_dataset(args.cache)
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    column = 'obs_fused' if args.input == 'fused' else 'obs'
    obs_rms, state_rms = fitted_statistics(dataset, column)

    device = torch.device(args.device)
    n_agents = int(dataset['num_cameras'])
    action_dim = dataset['actions'].shape[-1]
    msg_dim = config['env']['msg_dim']

    steps = args.max_episode_steps + 1
    stacked = stack_history(obs_rms.normalize(dataset[column]), args.history, steps)

    # Cloning is trained without a message input, and evaluation runs with the
    # channel off, so the two never disagree about what the actor is reading.
    # The previous command is a real actor input, so its ablation is a column of
    # zeros rather than a narrower network.
    previous = (
        previous_actions(dataset, steps)
        if args.prev_action
        else np.zeros_like(dataset['actions'])
    )

    obs = sequence_view(stacked, steps, device)
    prev = sequence_view(previous, steps, device)
    actions = sequence_view(dataset['actions'], steps, device)

    # Split by camera-episode, not by step: a sequence has to stay whole.
    train_index, test_index = split(obs.shape[0], seed=args.seed)
    train_index = torch.as_tensor(train_index, device=device)
    test_index = torch.as_tensor(test_index, device=device)

    torch.manual_seed(args.seed)
    policy = CommunicativeMAPPO(
        obs_dim=stacked.shape[-1],
        msg_dim=msg_dim,
        action_dim=action_dim,
        state_dim=dataset['states'].shape[-1],
        hidden_dim=config['policy']['hidden_dim'],
        device=args.device,
    )
    print(
        f'input: {column} x{args.history} | {obs.shape[0]} camera-episodes of {steps} steps | '
        f'greedy demonstrator coverage '
        f'{dataset["coverage"].mean():.4f} +- {dataset["coverage"].std():.4f}'
    )

    for epoch in range(args.epochs):
        loss = fit_actor(
            policy.actor,
            obs[train_index],
            prev[train_index],
            actions[train_index],
            action_dim,
            msg_dim,
            epochs=1,
            lr=config['policy']['actor_lr'],
            device=device,
            seed=args.seed + epoch,
        )
        held_out = sequence_error(
            policy.actor,
            obs[test_index],
            prev[test_index],
            actions[test_index],
            action_dim,
            msg_dim,
            device,
        )
        print(f'  epoch {epoch + 1:>3}: last step MSE {loss:.5f} | held out {held_out:.5f}')

    policy.actor.eval()

    # The same weights under both normalization regimes: whether the running
    # statistics keep updating during evaluation is then the only difference
    # between the two numbers, with no training noise in between.
    measured = {}
    for updating in (False, True):
        env = MATEEnv(
            scenario=args.scenario,
            max_episode_steps=args.max_episode_steps,
            msg_dim=msg_dim,
            camera_comm=False,
            frame_skip=config['env']['frame_skip'],
            seed=args.seed,
            shared_fov=args.input == 'fused',
            update_statistics=updating,
        )
        env.obs_rms = copy.deepcopy(obs_rms)
        env.state_rms = copy.deepcopy(state_rms)
        measured['updating' if updating else 'frozen'] = evaluate_rollout(
            env, policy, args.episodes, action_dim, args.prev_action, args.history
        )
        env.close()

    metrics = measured['frozen']
    os.makedirs(os.path.dirname(args.clone_path) or '.', exist_ok=True)
    torch.save(
        {
            'actor': policy.actor.state_dict(),
            'obs_rms': obs_rms.state_dict(),
            'state_rms': state_rms.state_dict(),
            'input': column,
            'history': args.history,
            'prev_action': args.prev_action,
            'action_dim': action_dim,
            'msg_dim': msg_dim,
            'hidden_dim': config['policy']['hidden_dim'],
        },
        args.clone_path,
    )
    print()
    for name, result in measured.items():
        print(
            f'cloned actor ({name} statistics): coverage {result["eval/coverage_rate"]:.4f} '
            f'+- {result["eval/coverage_rate_std"]:.4f} over {args.episodes} episodes'
        )
    print(f'greedy demonstrator: {dataset["coverage"].mean():.4f}')


@torch.no_grad()
def evaluate_rollout(env, policy, episodes, action_dim, feed_previous, history=1):
    """Roll the cloned actor out, with or without its own last command.

    ``evaluate_policy`` always feeds the previous command back, which is the
    trained behaviour; this variant can zero that input instead, which is the
    ablation the clone was fitted under.
    """

    from evaluate import weighted_mean

    coverages = []
    for _ in range(episodes):
        current = env.reset()
        previous = np.zeros((env.n_agents, action_dim), dtype=np.float32)
        messages = np.zeros((env.n_agents, env.msg_dim), dtype=np.float32)
        hidden = policy.actor.initial_state(env.n_agents, policy.device)
        window = []

        done = False
        coverage = []
        while not done:
            features = push_history(window, current['obs'], history)
            distribution, hidden = policy.actor.distribution(
                torch.as_tensor(features, dtype=torch.float32, device=policy.device),
                torch.as_tensor(messages, device=policy.device),
                torch.as_tensor(previous, device=policy.device),
                hidden,
            )
            mean = distribution.mean
            action = mean[:, :action_dim].clamp(-1.0, 1.0).cpu().numpy()

            current, _, done, info = env.step(action, messages)
            if feed_previous:
                previous = action.astype(np.float32)
            coverage.append((info['coverage_rate'], info['env_steps']))

        coverages.append(weighted_mean(coverage))

    return {
        'eval/coverage_rate': float(np.mean(coverages)),
        'eval/coverage_rate_std': float(np.std(coverages)),
    }


def advisor_rollout(
    env, actor, obs_rms, episodes, base_seed, action_dim, msg_dim, device, fused,
    feed_previous, history=1,
):
    """Drive with ``actor`` while GreedyCameraAgent rides along, labelling.

    The advisor observes and communicates exactly as it would if it were
    driving; only its command is discarded and recorded.  One rollout therefore
    yields both the covariate-shift measurement and the relabelled pairs DAgger
    trains on.
    """

    import mate
    from gymnasium.utils import seeding
    from mate.agents import GreedyCameraAgent

    unwrapped = env.unwrapped
    counts = (unwrapped.num_cameras, unwrapped.num_targets, unwrapped.num_obstacles)
    low = unwrapped.camera_action_space.low.astype(np.float64)
    high = unwrapped.camera_action_space.high.astype(np.float64)

    messages = torch.zeros(counts[0], msg_dim, device=device)
    observations, previouses, advices, errors, coverages = [], [], [], [], []

    for episode in range(episodes):
        prototype = GreedyCameraAgent()
        prototype.seed(base_seed + episode)
        advisors = prototype.spawn(counts[0])

        for offset, target_agent in enumerate(env.opponent_agents_ordered):
            target_agent._np_random, _ = seeding.np_random(
                base_seed + episode + 1000 * (offset + 1)
            )
        observation, _ = env.reset(seed=base_seed + episode)
        if fused:
            observation = fuse_camera_observations(observation, *counts)
        mate.group_reset(advisors, observation)

        previous = np.zeros((counts[0], action_dim), dtype=np.float32)
        hidden = actor.initial_state(counts[0], device)
        window = []
        infos = None
        coverage = []
        done = False
        while not done:
            advice = np.asarray(
                mate.group_step(unwrapped, advisors, observation, infos), dtype=np.float64
            )
            advice = (2.0 * (advice - low) / (high - low) - 1.0).astype(np.float32)

            features = push_history(
                window, obs_rms.normalize(np.asarray(observation, dtype=np.float64)), history
            )
            with torch.no_grad():
                distribution, hidden = actor.distribution(
                    torch.as_tensor(features, dtype=torch.float32, device=device),
                    messages,
                    torch.as_tensor(previous, device=device),
                    hidden,
                )
                mean = distribution.mean
            action = mean[:, :action_dim].clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

            observations.append(features.astype(np.float32))
            previouses.append(previous.copy())
            advices.append(advice)
            errors.append(float(np.mean((action - advice) ** 2)))

            if feed_previous:
                previous = action.copy()

            scaled = low + 0.5 * (action.astype(np.float64) + 1.0) * (high - low)
            observation, _, terminated, truncated, infos = env.step(scaled)
            if fused:
                observation = fuse_camera_observations(observation, *counts)
            done = bool(terminated) or bool(truncated)
            coverage.append(float(infos[0]['coverage_rate']))

        coverages.append(float(np.mean(coverage)))

    return {
        'obs': np.stack(observations),
        'prev_actions': np.stack(previouses),
        'advice': np.stack(advices),
        'action_mse': float(np.mean(errors)),
        'coverage': float(np.mean(coverages)),
        'coverage_std': float(np.std(coverages)),
    }


def advisor_environment(scenario, max_episode_steps):
    import mate
    from mate.agents import GreedyTargetAgent

    base_env = mate.make_environment(config=scenario, max_episode_steps=max_episode_steps)
    return mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())


def sequence_view(array, episode_length, device):
    """``(N, agents, dim)`` in decision order -> ``(agents * episodes, steps, dim)``.

    The GRU has to see an episode in order, and every camera is an independent
    sequence, so the agent axis folds into the batch and time stays intact.
    """

    episodes = array.shape[0] // episode_length
    assert episodes * episode_length == array.shape[0], (
        f'{array.shape[0]} decisions do not divide into episodes of {episode_length}'
    )

    shaped = array.reshape(episodes, episode_length, *array.shape[1:])
    shaped = np.swapaxes(shaped, 1, 2).reshape(-1, episode_length, array.shape[-1])
    return torch.as_tensor(shaped, dtype=torch.float32, device=device)


def fit_actor(actor, obs, prev, targets, action_dim, msg_dim, epochs, lr, device, seed=0):
    """Regress the actor mean onto the advisor commands, in episode order.

    One optimizer step per timestep with the belief detached between them: the
    same one-step-of-backpropagation scheme the PPO update uses, so the actor is
    trained the way it is later scored.
    """

    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    rows, steps = obs.shape[0], obs.shape[1]
    messages = torch.zeros(rows, msg_dim, device=device)
    torch.manual_seed(seed)
    loss = torch.zeros(())

    for _ in range(epochs):
        hidden = actor.initial_state(rows, device)
        for step in range(steps):
            distribution, hidden = actor.distribution(
                obs[:, step], messages, prev[:, step], hidden
            )
            loss = F.mse_loss(distribution.mean[:, :action_dim], targets[:, step])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            hidden = hidden.detach()
    return float(loss.item())


@torch.no_grad()
def sequence_error(actor, obs, prev, targets, action_dim, msg_dim, device):
    """Mean squared command error over held-out episodes, belief carried."""

    rows, steps = obs.shape[0], obs.shape[1]
    messages = torch.zeros(rows, msg_dim, device=device)
    hidden = actor.initial_state(rows, device)
    total = 0.0

    for step in range(steps):
        distribution, hidden = actor.distribution(obs[:, step], messages, prev[:, step], hidden)
        total += F.mse_loss(distribution.mean[:, :action_dim], targets[:, step]).item()
    return total / steps


def run_shift(args):
    """How far the cloned actor drifts from the expert once it drives.

    Cloning is fitted on the states the expert visits.  Held-out error on the
    expert's own states versus error on the clone's states is the size of the
    covariate shift, and it separates "the actor cannot represent the expert"
    from "the actor never saw the states it steers itself into".
    """

    ensure_mate_importable()
    scenario = resolve_scenario(args.scenario)

    from algorithms.communicative_mappo import CommunicativeActor

    saved = torch.load(args.clone_path, map_location=args.device, weights_only=False)
    obs_rms = RunningMeanStd((1,))
    obs_rms.load_state_dict(saved['obs_rms'])

    action_dim, msg_dim = saved['action_dim'], saved['msg_dim']
    history = saved.get('history', 1)
    actor = CommunicativeActor(
        obs_dim=obs_rms.mean.shape[0] * history,
        msg_dim=msg_dim,
        action_dim=action_dim,
        hidden_dim=saved['hidden_dim'],
    ).to(args.device)
    actor.load_state_dict(saved['actor'])
    actor.eval()

    env = advisor_environment(scenario, args.max_episode_steps)
    result = advisor_rollout(
        env,
        actor,
        obs_rms,
        args.episodes,
        args.seed,
        action_dim,
        msg_dim,
        args.device,
        fused=saved['input'] == 'obs_fused',
        feed_previous=saved['prev_action'],
        history=history,
    )
    env.close()

    print(f'clone {os.path.basename(args.clone_path)} driving for {args.episodes} episodes:')
    print(
        f"  action MSE against the advisor, on the clone's own states: "
        f"{result['action_mse']:.5f}"
    )
    print(f"  coverage while driving: {result['coverage']:.4f} +- {result['coverage_std']:.4f}")


def run_dagger(args):
    """Train the actor on the states it actually reaches, not the expert's.

    Round 0 is plain cloning.  Every later round drives the current actor, has
    the advisor label each state it lands in, adds those pairs to the training
    set and refits.  Cloning measures what a single-step actor does when it
    never sees its own mistakes; this measures the same actor once it does.  A
    curve that climbs toward the demonstrator says the architecture is
    sufficient and the whole gap is the state distribution the policy trains
    on -- which is what imagination is supposed to supply.
    """

    from algorithms.communicative_mappo import CommunicativeActor

    dataset = load_dataset(args.cache)
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    obs_rms, _ = fitted_statistics(dataset)

    ensure_mate_importable()
    scenario = resolve_scenario(args.scenario)

    device = torch.device(args.device)
    action_dim = dataset['actions'].shape[-1]
    msg_dim = config['env']['msg_dim']

    steps = args.max_episode_steps + 1
    stacked = stack_history(obs_rms.normalize(dataset['obs']), args.history, steps)

    obs = sequence_view(stacked, steps, device)
    prev = sequence_view(previous_actions(dataset, steps), steps, device)
    targets = sequence_view(dataset['actions'], steps, device)

    torch.manual_seed(args.seed)
    actor = CommunicativeActor(
        obs_dim=stacked.shape[-1],
        msg_dim=msg_dim,
        action_dim=action_dim,
        hidden_dim=config['policy']['hidden_dim'],
    ).to(device)

    env = advisor_environment(scenario, args.max_episode_steps)
    print(
        f'demonstrator {dataset["coverage"].mean():.4f} | {args.rounds} rounds | '
        f'history {args.history}'
    )

    for round_index in range(args.rounds):
        loss = fit_actor(
            actor,
            obs,
            prev,
            targets,
            action_dim,
            msg_dim,
            args.epochs,
            config['policy']['actor_lr'],
            device,
            seed=args.seed + round_index,
        )
        actor.eval()
        result = advisor_rollout(
            env,
            actor,
            obs_rms,
            args.episodes,
            args.seed + 100 * round_index,
            action_dim,
            msg_dim,
            device,
            fused=False,
            feed_previous=True,
            history=args.history,
        )
        actor.train()

        print(
            f'  round {round_index}: {obs.shape[0]:>5} camera-episodes | '
            f'last step loss {loss:.5f} | '
            f'action MSE on own states {result["action_mse"]:.5f} | '
            f'coverage {result["coverage"]:.4f} +- {result["coverage_std"]:.4f}'
        )

        obs = torch.cat([obs, sequence_view(result['obs'], steps, device)])
        prev = torch.cat([prev, sequence_view(result['prev_actions'], steps, device)])
        targets = torch.cat([targets, sequence_view(result['advice'], steps, device)])

    env.close()



def run_noise(args):
    """How much action precision the coverage target actually demands.

    Cloning reached a held-out action RMSE and still lost most of the
    demonstrator's coverage, which only makes sense if the task punishes small
    per-step control error in closed loop.  This perturbs the demonstrator
    itself by a known sigma in the same [-1, 1] units and reads off the curve,
    turning "the policy is not accurate enough" into a number a policy can be
    held to.
    """

    from phase0 import run_rule_agent

    ensure_mate_importable()
    scenario = resolve_scenario(args.scenario)

    print(f'{scenario} | greedy with Gaussian action noise | {args.episodes} episodes')
    for sigma in args.sigmas:
        result = run_rule_agent(
            scenario,
            'greedy',
            args.episodes,
            args.max_episode_steps,
            args.seed,
            shared_fov=False,
            communicate=True,
            action_noise=sigma,
        )
        print(
            f'  sigma {sigma:.2f} (RMSE {sigma:.2f} of a 2.0-wide action range): '
            f'coverage {result["coverage_rate"]:.4f} +- {result["coverage_rate_std"]:.4f}'
        )


# ------------------------------------------------------------------------ main


def run_dataset(args):
    ensure_mate_importable()
    dataset = collect_greedy_dataset(
        resolve_scenario(args.scenario), args.episodes, args.max_episode_steps, args.seed
    )
    os.makedirs(os.path.dirname(args.cache) or '.', exist_ok=True)
    np.savez_compressed(args.cache, **dataset)
    print(
        f'{dataset["obs"].shape[0]} decisions from {args.episodes} greedy episodes | '
        f'coverage {dataset["coverage"].mean():.4f} +- {dataset["coverage"].std():.4f} | '
        f'saved to {args.cache}'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        choices=('dataset', 'tokenizer', 'clone', 'noise', 'shift', 'dagger'),
    )
    parser.add_argument('--scenario', type=str, default='MATE-4v8-9')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--updates', type=int, default=4000, help='tokenizer gradient steps')
    parser.add_argument(
        '--tokens',
        type=int,
        nargs='*',
        default=[16, 64],
        help='token budgets to compare; the first is the configured one',
    )
    parser.add_argument('--epochs', type=int, default=20, help='cloning epochs')
    parser.add_argument(
        '--clone-path',
        type=str,
        default=os.path.join('runs', 'diagnose', 'clone.pt'),
        help='where `clone` saves its actor and where `shift` reads it from',
    )
    parser.add_argument(
        '--sigmas',
        type=float,
        nargs='*',
        default=[0.0, 0.1, 0.2, 0.3, 0.5],
        help='action-noise levels for the sensitivity sweep',
    )
    parser.add_argument('--rounds', type=int, default=6, help='DAgger rounds')
    parser.add_argument(
        '--history',
        type=int,
        default=1,
        help='how many past observations the actor reads, newest first',
    )
    parser.add_argument(
        '--prev-action',
        action='store_true',
        help="append the camera's own previous command to the cloned actor's input",
    )
    parser.add_argument(
        '--input',
        choices=('raw', 'fused'),
        default='raw',
        help="observation the cloned actor reads: the camera's own, or the team union",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    {
        'dataset': run_dataset,
        'tokenizer': run_tokenizer,
        'clone': run_clone,
        'noise': run_noise,
        'shift': run_shift,
        'dagger': run_dagger,
    }[args.command](args)


if __name__ == '__main__':
    main()
