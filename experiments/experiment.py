"""Comparing pink action noise with the default noise on SAC."""

import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
import shimmy
import numpy as np
import torch
from pink.lpnrl import LowPassNoiseDist
from pink.sb3 import PinkNoiseDist
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from experiment_launcher import single_experiment, run_experiment

@single_experiment
def experiment(
    noise_type: str = "pink",
    env_name: str = "HalfCheetah-v4",
    learning_starts: int = 10_000,
    total_timesteps: int = 1_000_000,
    cutoff: float = 5.0,
    order: int = 1,
    seed: int = 0,
    debug: bool = True,
    #debug: bool = False,
    results_dir: str = "results",
    ):
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    config = dict(
        alg=alg,
        noise_type=noise_type,
        env_name=env_name,
        learning_starts=learning_starts,
        cutoff=cutoff,
        order=order,
    )

    # Initialize environment
    try:
        env = gym.make(config["env_name"])
        seq_len = env._max_episode_steps
    except:
        env = suite.load(*config["env_name"].split("-"))
        seq_len = int(env._step_limit)
        env = FlattenObservation(shimmy.DmControlCompatibilityV0(env))
    env = Monitor(env)
    action_dim = env.action_space.shape[-1]

    #group_name = f"{config['env_name']}_{config['noise_type']}_ls{config['learning_starts']}"
    group_name = f"{config['noise_type']}"
    if config['noise_type'] == "lowpass":
        group_name = group_name + f"_fc{config['cutoff']}_o{config['order']}"
    run_name = group_name + f"_seed{seed}"

    run = wandb.init(
        project="LPRL_" + alg + "_" + env_name,
        group=group_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        save_code=True,  # optional
        mode="online" if not debug else "disabled",
    )

    # Initialize agents
    if alg == "sac":
        model = SAC("MlpPolicy", env, seed=seed, verbose=1, tensorboard_log=f"runs/{run.id}", learning_starts=config["learning_starts"])
    elif alg == "td3":
        model = TD3("MlpPolicy", env, seed=seed, verbose=1, tensorboard_log=f"runs/{run.id}", learning_starts=config["learning_starts"])

    if config["noise_type"] == "pink":
        model.actor.action_dist = PinkNoiseDist(seq_len, action_dim, rng=rng)
    elif config["noise_type"] == "lowpass":
        dt = env.unwrapped.dt
        model.actor.action_dist = LowPassNoiseDist(cutoff=config["cutoff"], order=config["order"], sampling_freq=1./dt,
                                                   seq_len=seq_len, action_dim=action_dim, rng=rng)

    # Train agents
    model.learn(total_timesteps=total_timesteps, progress_bar=True,
                callback=WandbCallback(
                    model_save_freq=int(0.1 * total_timesteps),
                    model_save_path=f"models/{run.id}",
                    verbose=2,),
                )

    run.finish()

if __name__ == "__main__":
    run_experiment(experiment)