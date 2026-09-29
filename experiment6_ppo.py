#!/usr/bin/env python3
"""實驗 6：用 PPO 學習實驗 5 的頭部追蹤控制。

本檔與 experiment5_head_env.py 放同一資料夾。保留實驗 5 的 XML、reward、
action 與 observation，不以手寫控制器示範資料訓練。

安裝（現有 MuJoCo/Gymnasium 不需更新）：
python -m pip install "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "stable-baselines3==2.9.0"
Python >=3.10；使用 CPU 即可。本程式無需 TensorBoard。

python experiment6_ppo.py train --steps 100000 --out runs/head_ppo_01
python experiment6_ppo.py evaluate --run runs/head_ppo_01 --episodes 10
python experiment6_ppo.py evaluate --run runs/head_ppo_01 --episodes 1 --render

CPU、單環境、無觀測/獎勵正規化。actor/critic 各兩層 64 單元 Tanh。
訓練 Gaussian action 有探索；評估 deterministic=True，再由 SB3 截限 action。
預設評估 seed 10000 起，與訓練初始 seed 42 分開；不是新頻率泛化測試。
initial.zip 與 final.zip 的 deterministic 行為都會比較，外加三種 baseline。
訓練總步數會向上取整到 1024 的倍數。模型格式是 SB3 ZIP，並非 ONNX。
--render 只顯示 final 策略；保留已驗證的一秒關閉緩衝，非正式 thread join。
此為教學用 SB3 PPO，不是 microduck 官方 mjlab/rsl_rl 訓練入口。
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import experiment5_head_env as source


def source_hash():
    return hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()


def train(args):
    if args.steps < 1:
        raise ValueError("steps 必須 > 0")
    # 不覆寫舊實驗，請每次換新的資料夾名稱。
    args.out.mkdir(parents=True, exist_ok=False)
    env = source.HeadTrackingEnv(frequency=.5, duration=10.)
    check_env(env, warn=True)
    env = Monitor(env, str(args.out / "train.monitor.csv"))
    config = dict(frequency=.5, duration=10., seed=args.seed,
                  requested_steps=args.steps, env_sha256=source_hash(),
                  versions={p: importlib.metadata.version(p) for p in
                            ["stable-baselines3", "torch", "mujoco", "gymnasium", "numpy"]})
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        # [A] 建立尚未訓練的 actor 與 critic。
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024,
                    batch_size=64, n_epochs=10, gamma=.99, gae_lambda=.95,
                    clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                    seed=args.seed, device="cpu", verbose=1)
        model.set_logger(configure(str(args.out), ["stdout", "csv"]))
        model.save(args.out / "initial")
        # [B] 真正發生學習的地方：蒐集 rollout，估計 advantage，更新網路。
        model.learn(total_timesteps=args.steps,
                    callback=CheckpointCallback(save_freq=10000,
                                                save_path=str(args.out / "checkpoints"),
                                                name_prefix="ppo"))
        model.save(args.out / "final")
        config["actual_steps"] = model.num_timesteps
        (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Saved: {args.out / 'final.zip'}", flush=True)
    finally:
        env.close()


def evaluate(args):
    if args.episodes < 1:
        raise ValueError("episodes 必須 > 0")
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["env_sha256"] != source_hash():
        raise RuntimeError("實驗 5 原始碼與訓練時不同；請還原同一版本再比較")
    models = {k: PPO.load(args.run / f"{k}.zip", device="cpu") for k in ["initial", "final"]}
    output = args.run / ("eval_" + str(time.time_ns()))
    output.mkdir()
    summaries = []
    for mode in ["zero", "direct", "feedback", "initial", "final"]:
        show = args.render and mode == "final"
        env = source.HeadTrackingEnv(config["frequency"], config["duration"],
                                     "human" if show else None)
        viewer_used = False
        errors, rewards, tail_errors = [], [], []
        failures = 0
        try:
            with (output / f"{mode}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = None
                for episode in range(args.episodes):
                    # [C] 五種控制器使用完全相同的評估 seeds。
                    obs, _ = env.reset(seed=args.eval_seed + episode)
                    for step in range(env.max_steps):
                        start = time.perf_counter()
                        if mode in models:
                            action, _ = models[mode].predict(obs, deterministic=True)
                        else:
                            action = source.baseline_action(obs, env, mode)
                        obs, reward, terminated, truncated, info = env.step(action)
                        row = dict(episode=episode, seed=args.eval_seed + episode,
                                   step=step + 1, **info)
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        errors.append(info["error_deg"])
                        rewards.append(reward)
                        if info["time_s"] >= 2.:
                            tail_errors.append(info["error_deg"])
                        if show:
                            viewer_used = True
                            env.render()
                            if not env.viewer.is_running():
                                raise KeyboardInterrupt("Viewer closed; partial CSV retained")
                            time.sleep(max(0, env.dt - (time.perf_counter() - start)))
                        if terminated or truncated:
                            failures += int(terminated)
                            break
            result = dict(controller=mode, episodes=args.episodes, samples=len(errors),
                          rmse_deg=float(np.sqrt(np.mean(np.square(errors)))),
                          rmse_after_2s_deg=(float(np.sqrt(np.mean(np.square(tail_errors))))
                                            if tail_errors else None),
                          mean_reward=float(np.mean(rewards)), failed_episodes=failures)
            summaries.append(result)
            print(f"{mode:8s} RMSE={result['rmse_deg']:.3f} deg "
                  f"reward={result['mean_reward']:.4f} failures={failures}", flush=True)
        finally:
            env.close()
            if viewer_used:
                time.sleep(1.)  # 相容舊版 experiment5 的關閉緩衝；新版可能再等一秒。
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Evaluation: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=100000)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--out", type=Path, default=Path("runs/head_ppo_01"))
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run", type=Path, default=Path("runs/head_ppo_01"))
    ev.add_argument("--episodes", type=int, default=10)
    ev.add_argument("--eval-seed", type=int, default=10000)
    ev.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
