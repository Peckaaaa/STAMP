# DIMA_base — the baselines the unified pipeline replaced

Archive of the modules that were in the training loop before the repository
moved to a single **diffusion-driven trajectory planning** pipeline. Nothing
here is imported by `train.py`, `evaluate.py`, `phase0.py` or
`check_pipeline.py` any more.

They are kept for two reasons: the numbers they produced are the reference
points the current work is measured against, and the design decisions inside
them are documented in their own docstrings — several were arrived at by
measurement and are worth reading before anyone re-derives them.

## What is here

| File | What it was |
|---|---|
| `src/algorithms/communicative_mappo.py` | Actor–critic MAPPO. The actor ended as `pi(a | ô, b, p̂, a_prev, h)` with a GRU belief; PPO on flat minibatches, stored recurrent state as an input rather than backpropagation through time |
| `src/algorithms/onpolicy_collector.py` | Real-environment rollouts shaped for that PPO update — the model-free floor |
| `src/algorithms/world_model_trainer.py` | Owned the state-space world model: FSQ tokenizer, categorical diffusion, reward/termination model, and the H-step imagination MAPPO trained on |
| `src/algorithms/replay_buffer.py` | Ring buffer for that stack: states, observations, beliefs, actions, previous actions, recurrent states, target positions |
| `src/models/state_autoencoder.py` | FSQ tokenizer of the global state plus every camera's belief |
| `src/models/categorical_diffusion.py` | D3PM over those tokens: absorbing or uniform corruption, MaskGIT-style confidence unmasking at sampling time |
| `src/models/transformer_denoiser.py` | Joint-attention denoiser — every camera one token, no agent ordering, unlike the sequential per-agent denoising of the original DIMA |
| `src/models/reward_termination.py` | minGPT-style reward and termination heads over token sequences |
| `src/models/fsq_quantizer.py` | Finite scalar quantization |
| `diagnose.py` | The instrument that located the bottleneck: tokenizer capacity, behaviour cloning against `GreedyCameraAgent`, action-noise sensitivity, covariate shift, DAgger, observation-window and recurrence ablations |

## The numbers they produced

`MATE-4v8-9`, 20-episode protocol, single seed each, 150k environment steps
unless stated:

| Configuration | Coverage |
|---|---|
| MAPPO, model-free, stateless actor | 0.3405 |
| MAPPO, model-free, plus previous command | 0.3276 |
| MAPPO, model-free, plus GRU belief | 0.2908 |
| State-space world model + MAPPO (earlier architecture) | 0.4035 |

And what `diagnose.py` established, which is why the pipeline changed:

- The FSQ tokenizer was **not** the bottleneck: 144 bits reconstructed the
  global state to a held-out MSE of 0.0078, and quadrupling the budget improved
  visible-target position error only from 133 to 109 map units.
- Control precision was **not** the bottleneck: the greedy rule agent loses only
  0.018 coverage to Gaussian action noise of σ = 0.5 on a 2.0-wide action box.
- Actor memory was not enough either: a cloned actor's command error against the
  reference agent was 0.069 on the expert's own states but 1.27 on the states it
  reached itself, and neither a 16-step observation window (1.20) nor a GRU
  (1.16) closed that, nor did 56k DAgger labels collected on those very states.

That is what moved the work from "train a better actor" to "plan online against
a generative trajectory model".

## These files do not run as they stand

The environment wrapper changed underneath them. `MATEEnv` now returns
`belief`, `peer_intent`, `camera_states` and `target_positions`, takes
`step(actions, intents)`, and no longer produces the `messages` these modules
expect; the buffer API and the actor signature changed with it.

The baseline numbers above were produced by successive states of these files
during development, not all of which were committed, so this snapshot is a
reference copy rather than a reproduction recipe. To re-run a baseline, restore
the matching commit — do not try to re-wire these modules into the current
`train.py`.
