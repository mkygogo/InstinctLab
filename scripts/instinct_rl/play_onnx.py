"""Script to play an ONNX policy in Isaac Lab."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an ONNX agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
#parser.add_argument("--load_run", type=str, required=True, help="Absolute path to the root checkpoint folder (e.g. stand_onboard)")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.tasks.parkour.scripts.onnxer import load_parkour_onnx_model

def main():
    """Play with ONNX agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # 1. 设置 ONNX 模型目录 (自动寻找 exported 子文件夹)
    onnx_dir = os.path.join(args_cli.load_run, "exported")
    if not os.path.exists(onnx_dir):
        raise FileNotFoundError(f"[ERROR] ONNX directory not found at: {onnx_dir}")
    print(f"[INFO] Successfully targeted ONNX model from: {onnx_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.load_run, "videos"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "onnx_play",
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl (这一步会把字典obs压平为一维张量)
    env = InstinctRlVecEnvWrapper(env)

    # 2. 核心：定义 ONNX 的观测空间切片逻辑
    # 根据日志，本体感觉长度 = 24(ang_vel) + 24(gravity) + 24(cmd) + 232(pos) + 232(vel) + 232(action) = 768
    proprio_len = 768
    depth_shape = (8, 18, 32)

    print("[INFO] Loading ONNX Inference Sessions into GPU...")
    # 3. 加载官方提供的 ONNX 适配器
    policy = load_parkour_onnx_model(
        model_dir=onnx_dir,
        get_subobs_func=lambda obs: obs[:, proprio_len:],  # 取出最后的 8*18*32 个像素特征
        depth_shape=depth_shape,
        proprio_slice=slice(0, proprio_len)  # 取出前面 768 个本体感觉特征
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    print("[INFO] Starting ONNX Simulation Loop...")
    
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # agent stepping (ONNX policy 会自动处理 numpy 和 tensor 的转换)
            actions = policy(obs)
            
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            
        timestep += 1

        if args_cli.video:
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()