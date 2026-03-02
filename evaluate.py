"""
evaluate.py -- evaluate a trained PPO agent and record a video.

Usage examples
--------------
# Evaluate and record video for a specific run
python evaluate.py --run_dir runs/dense_20250101_120000

# More episodes, custom output path
python evaluate.py --run_dir runs/dense_20250101_120000 --n_episodes 20 --video_out demo.mp4
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.walker_env import Walker2DEnv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Walker2D agent")
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the run directory (must contain final_model.zip and vecnorm.pkl)."
    )
    p.add_argument(
        "--n_episodes", type=int, default=20,
        help="Number of evaluation episodes."
    )
    p.add_argument(
        "--video_out", type=str, default=None,
        help="Path for the output video (e.g. demo.mp4). Set to 'none' to skip."
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Show a reward-per-episode bar chart after evaluation."
    )
    return p.parse_args()


def record_episode(model, env, vecnorm):
    """Run one episode, capture frames, return (total_reward, frames)."""
    frames = []
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        frame = env.envs[0].render()
        if frame is not None:
            frames.append(frame)
    return total_reward, frames


def main():
    args = parse_args()

    model_path = os.path.join(args.run_dir, "final_model.zip")
    vecnorm_path = os.path.join(args.run_dir, "vecnorm.pkl")

    if not os.path.exists(model_path):
        # Try best model fallback
        best_path = os.path.join(args.run_dir, "best_model", "best_model.zip")
        if os.path.exists(best_path):
            model_path = best_path
            print(f"Note: final_model.zip not found, using best_model.zip")
        else:
            raise FileNotFoundError(f"No model found in {args.run_dir}")

    record_video = (
        args.video_out is not None
        and args.video_out.lower() != "none"
    )

    render_mode = "rgb_array" if record_video else None

    # Build evaluation environment
    def make_eval_env():
        return Walker2DEnv(render_mode=render_mode)

    raw_env = DummyVecEnv([make_eval_env])

    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, raw_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print("Loaded VecNormalize statistics.")
    else:
        eval_env = raw_env
        print("Warning: vecnorm.pkl not found. Running without observation normalisation.")

    model = PPO.load(model_path, env=eval_env)

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    SUCCESS_DIST = 3.0   # metres

    print(f"\nEvaluating {args.n_episodes} episodes...")
    all_frames = []

    for ep in range(args.n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False
        ep_frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            if record_video and ep == 0:
                frame = eval_env.envs[0].render()
                if frame is not None:
                    ep_frames.append(frame)

        # Check walk success: torso moved >= 3 m in x direction
        torso_x = info[0].get("torso_pos", [0, 0, 0])[0]
        success = torso_x >= SUCCESS_DIST

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        if success:
            success_count += 1

        print(
            f"  Episode {ep + 1:3d}: reward={ep_reward:8.1f}  "
            f"length={ep_len:5d}  "
            f"torso_x={torso_x:5.1f} m  "
            f"{'SUCCESS' if success else 'fell'}"
        )

        if ep == 0:
            all_frames = ep_frames

    # Summary statistics
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    success_rate = success_count / args.n_episodes * 100

    print(f"\n{'='*55}")
    print(f"  Episodes:      {args.n_episodes}")
    print(f"  Mean reward:   {rewards.mean():.1f}  (+/- {rewards.std():.1f})")
    print(f"  Min / Max:     {rewards.min():.1f} / {rewards.max():.1f}")
    print(f"  Mean length:   {lengths.mean():.0f} steps")
    print(f"  Walk success:  {success_count}/{args.n_episodes}  ({success_rate:.0f}%)")
    print(f"{'='*55}")

    # Save video
    if record_video and all_frames:
        video_path = args.video_out or os.path.join(args.run_dir, "demo.mp4")
        try:
            import imageio
            imageio.mimwrite(video_path, all_frames, fps=60)
            print(f"\nVideo saved to: {video_path}")
        except ImportError:
            print("imageio not installed. Run: pip install imageio imageio-ffmpeg")

    # Optional plot
    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(1, args.n_episodes + 1), episode_rewards, color="steelblue")
        axes[0].axhline(rewards.mean(), color="red", linestyle="--", label=f"mean={rewards.mean():.0f}")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Episode Rewards")
        axes[0].legend()

        axes[1].bar(range(1, args.n_episodes + 1), episode_lengths, color="seagreen")
        axes[1].axhline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.0f}")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].set_title("Episode Lengths")
        axes[1].legend()

        plt.tight_layout()
        fig_path = os.path.join(args.run_dir, "eval_plot.png")
        plt.savefig(fig_path, dpi=120)
        print(f"Plot saved to: {fig_path}")
        plt.show()

    eval_env.close()


if __name__ == "__main__":
    main()
