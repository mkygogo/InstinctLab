"""Script to play an ONNX policy in Isaac Lab with Keyboard Hijack."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an ONNX agent with Keyboard Control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=3000)
parser.add_argument("--video_start_step", type=int, default=0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Instinct-Parkour-Target-Amp-G1-Play-v0")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.tasks.parkour.scripts.onnxer import load_parkour_onnx_model

def main():
    if args_cli.load_run is None:
        raise ValueError("[ERROR] Please specify the ONNX model directory using --load_run")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    onnx_dir = os.path.join(args_cli.load_run, "exported")
    print(f"[INFO] Targeted ONNX model from: {onnx_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = InstinctRlVecEnvWrapper(env)

    # 观测空间切片参数
    proprio_len = 768
    depth_shape = (8, 18, 32)

    policy = load_parkour_onnx_model(
        model_dir=onnx_dir,
        get_subobs_func=lambda obs: obs[:, proprio_len:],
        depth_shape=depth_shape,
        proprio_slice=slice(0, proprio_len)
    )

    # ==========================================
    # ⚠️ 新增：初始化键盘遥控器 (更新为 Se2Keyboard)
    # ==========================================
    print("[INFO] Initializing Keyboard Teleop...")
    
    # 1. 创建配置对象：里面什么参数都不传！(Isaac Lab 会自动加载底层的默认灵敏度和键位)
    teleop_cfg = Se2KeyboardCfg()
    
    # 2. 将配置对象老老实实地传给 cfg 参数
    teleop_interface = Se2Keyboard(cfg=teleop_cfg)
    
    # 3. 重置状态，激活底层事件监听
    teleop_interface.reset()
    
    print("========================================")
    print("🎮 键盘控制说明 (需选中 Isaac Sim 画面):")
    print("    方向键 : 前进 / 后退")
    print("========================================")

    obs, _ = env.get_observations()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # --- 1. 读取键盘输入 ---
            cmd_data = teleop_interface.advance()
            if isinstance(cmd_data, tuple):
                cmd_data = cmd_data[0]
                
            if isinstance(cmd_data, torch.Tensor):
                kb_cmd = cmd_data.to(device=env.device, dtype=torch.float32)
            else:
                kb_cmd = torch.tensor(cmd_data, device=env.device, dtype=torch.float32)
            
            # --- 2. 核心劫持：覆盖环境原有的自动速度指令 ---
            hijacked_cmd_history = kb_cmd.repeat(8)
            obs[:, 48:72] = hijacked_cmd_history

            # --- 3. 喂给跑酷大脑 ---
            actions = policy(obs)
            
            obs, rewards, dones, infos = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()