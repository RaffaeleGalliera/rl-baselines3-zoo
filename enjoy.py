import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_saved_hyperparams
from utils.callbacks import tqdm
from utils.exp_manager import ExperimentManager
from utils.load_from_hub import download_from_hub
from utils.utils import StoreDict, get_model_path

from envs.utils.constants import State
import time
import csv

from pathlib import Path

def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("-eps", "--n-episodes", help="number of episodes", default=10, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device=args.device, **kwargs)

    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(args.n_timesteps)
    if args.progress:
        generator = tqdm(generator)

    evaluation_folder = f"evaluation/{algo}/{args.exp_id}"
    Path(evaluation_folder).mkdir(parents=True, exist_ok=True)

    try:
        throughput_sum = 0
        goodput_sum = 0
        rtt_sum = 0
        retransmissions_sum = 0
        cwnd_sum = 0
        delay_sum = 0
        step_logs = []
        episodes_done = 0

        while episodes_done < args.n_episodes:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)

            episode_start = done

            for info in infos:
                if 'current_statistics' in info:
                    throughput_sum += info['current_statistics'][
                        State.THROUGHPUT]
                    goodput_sum += info['current_statistics'][
                        State.GOODPUT]
                    rtt_sum += info['current_statistics'][
                        State.LAST_RTT]
                    retransmissions_sum += info['current_statistics'][
                        State.RETRANSMISSIONS]
                    cwnd_sum += info['current_statistics'][
                        State.CURR_WINDOW_SIZE]
                    delay_sum += info['action_delay']

                    step_logger = {
                        "throughput_KB": info[
                            'current_statistics'][State.THROUGHPUT],
                        "goodput_KB": info[
                            'current_statistics'][State.GOODPUT],
                        "rtt_ms": info[
                            'current_statistics'][State.LAST_RTT],
                        "retransmissions": info[
                            'current_statistics'][State.RETRANSMISSIONS],
                        "current_window_size_KB": info[
                            'current_statistics'][State.CURR_WINDOW_SIZE],
                        'action': info['action'],
                        'action_delay_ms': info['action_delay'],
                        'rewards': info['reward'],
                        'timestamp': time.time_ns()
                    }
                    step_logs.append(step_logger)

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    episodes_done += 1
                    avg_episodic_throughput = throughput_sum / \
                                              info["episode"]["l"]
                    avg_episodic_goodput = goodput_sum / info["episode"][
                        "l"]
                    avg_episodic_rtt = rtt_sum / info["episode"]["l"]
                    avg_episodic_retransmissions = retransmissions_sum / \
                                                   info["episode"]["l"]
                    avg_window_size = cwnd_sum / info["episode"]["l"]
                    avg_delay = delay_sum / info["episode"]["l"]

                    episode_logger = {
                        "episodic_return": info["episode"]["r"],
                        "episodic_length": info["episode"]["l"],
                        "episodic_avg_throughput_KB":
                            avg_episodic_throughput,
                        "episodic_avg_goodput_KB":
                            avg_episodic_goodput,
                        "episodic_avg_rtt_ms": avg_episodic_rtt,
                        "episodic_avg_retransmissions": avg_episodic_retransmissions,
                        "total_retransmissions": retransmissions_sum,
                        "episodic_window_size_KB":
                            avg_window_size,
                        "avg_delay_ms": avg_delay,
                        "seconds_taken": time.time() - info['start_time']
                    }

                    print("Saving Steps Data to CSV")
                    try:
                        keys = step_logs[0].keys()
                        filename = f"{evaluation_folder}/step_logging_ep{episodes_done}.csv"
                        file_exists = os.path.isfile(filename)
                        with open(filename, 'a',
                                  newline='') as output_file:
                            dict_writer = csv.DictWriter(output_file, keys)
                            if not file_exists:
                                dict_writer.writeheader()
                            dict_writer.writerows(step_logs)

                    except IOError:
                        print("I/O error")

                    print("Saving AVG Data to CSV")

                    try:
                        keys = episode_logger.keys()
                        filename = f"{evaluation_folder}/episode_logger.csv"
                        file_exists = os.path.isfile(filename)

                        with open(filename, 'a+',newline='') as output_file:
                            dict_writer = csv.DictWriter(output_file, keys)
                            if file_exists:
                                dict_writer.writeheader()
                            dict_writer.writerow(episode_logger)

                    except IOError:
                        print("I/O error")

                    throughput_sum = 0
                    goodput_sum = 0
                    rtt_sum = 0
                    retransmissions_sum = 0
                    cwnd_sum = 0
                    delay_sum = 0
                    step_logs = []

                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    main()
