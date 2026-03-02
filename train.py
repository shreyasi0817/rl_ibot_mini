"""
train.py -- train a PPO agent on the Walker2D environment.

Usage examples
--------------
# Baseline training (dense reward, default hyperparameters)
python train.py

# Choose a different reward function
python train.py --reward sparse
python train.py --reward velocity_only
python train.py --reward heavy_energy

# Change hyperparameters for Week 3 ablation experiments
python train.py --lr 1e-3 --n_steps 4096 --net_arch "256 256" --total_steps 500000

# Full list of arguments
python train.py --help
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env.walker_env import Walker2DEnv
from env.reward_functions import get_reward_fn


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Walker2D")

    # Reward and output
    p.add_argument(
        "--reward", type=str, default="dense",
        choices=["sparse", "dense", "velocity_only", "heavy_energy"],
        help="Which reward function to use."
    )
    p.add_argument(
        "--run_name", type=str, default=None,
        help="Name for this run. Auto-generated if not given."
    )
    p.add_argument(
        "--out_dir", type=str, default="runs",
        help="Directory to save models and logs."
    )

    # Training length
    p.add_argument(
        "--total_steps", type=int, default=1_000_000,
        help="Total environment steps to train for."
    )
    p.add_argument(
        "--n_envs", type=int, default=4,
        help="Number of parallel environments."
    )

    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument(
        "--n_steps", type=int, default=2048,
        help="Steps per rollout per environment."
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size for PPO updates."
    )
    p.add_argument(
        "--n_epochs", type=int, default=10,
        help="Number of gradient epochs per rollout."
    )
    p.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor."
    )
    p.add_argument(
        "--gae_lambda", type=float, default=0.95,
        help="GAE lambda parameter."
    )
    p.add_argument(
        "--clip_range", type=float, default=0.2,
        help="PPO clip parameter epsilon."
    )
    p.add_argument(
        "--ent_coef", type=float, default=0.0,
        help="Entropy coefficient (encourages exploration)."
    )
    p.add_argument(
        "--net_arch", type=str, default="256 256",
        help="Hidden layer sizes e.g. '64 64' or '256 256'."
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Build a run name
    if args.run_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.reward}_{ts}"

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Run:          {args.run_name}")
    print(f"  Reward:       {args.reward}")
    print(f"  Total steps:  {args.total_steps:,}")
    print(f"  Envs:         {args.n_envs}")
    print(f"  LR:           {args.lr}")
    print(f"  n_steps:      {args.n_steps}")
    print(f"  Net arch:     {args.net_arch}")
    print(f"  Output dir:   {run_dir}")
    print(f"{'='*60}\n")

    reward_fn = get_reward_fn(args.reward)

    # Create vectorised training environment
    def make_env():
        return Walker2DEnv(reward_fn=reward_fn)

    train_env = make_vec_env(make_env, n_envs=args.n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Create vectorised evaluation environment (no reward normalisation)
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Network architecture
    net_arch = [int(x) for x in args.net_arch.split()]

    # Build PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=log_dir,
        verbose=1,
    )

    # Callbacks: checkpoint every 100k steps, eval every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="walker",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=max(50_000 // args.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo",
    )

    # Save final model and VecNormalize stats
    model_path = os.path.join(run_dir, "final_model")
    model.save(model_path)
    vecnorm_path = os.path.join(run_dir, "vecnorm.pkl")
    train_env.save(vecnorm_path)

    print(f"\nTraining complete.")
    print(f"  Model saved to:        {model_path}.zip")
    print(f"  VecNormalize saved to: {vecnorm_path}")
    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
