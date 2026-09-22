# learn/sac_config.py
"""SAC hyperparameters: a starting point modelled on published Donkey-sim +
VAE setups. net_arch must stay two equal hidden layers: the exported head
(learn/nets.py::PolicyHead, car/policy.py::Head) has exactly that shape,
and BC uses the same size."""

SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 7.3e-4,
    "buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.02,
    "ent_coef": "auto",
    # The sim runs in real time: gradient updates between env steps would
    # delay every command, so collect a whole episode, then do one gradient
    # step per collected step (-1) while the car is being reset.
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
    "learning_starts": 1000,
    # State-dependent exploration: smoother steering than per-step noise.
    # Also what makes the actor's mean clip to [-2, 2] (mirrored in the head).
    "use_sde": True,
    "sde_sample_freq": 16,
    "policy_kwargs": {"log_std_init": -2, "net_arch": [64, 64]},
}
