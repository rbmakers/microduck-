#!/usr/bin/env python3
"""實驗 7：固定 PPO 模型的頻率泛化測試（不訓練）。

與 experiment5_head_env.py 放在同一資料夾，使用實驗 6 的套件環境。
python experiment7_frequency_test.py --run runs/head_ppo_01
python experiment7_frequency_test.py --run runs/head_ppo_01 --frequencies 1.0 --episodes 1 --render

預設 0.25/0.5/0.75/1 Hz，10 回合，每回合 10 秒，前 2 秒視為暫態。
同一 seed 提供相同初始角度/相位；在每個頻率下三種控制器的命令完全相同。
頻率改變也會改變 command_velocity 的分布，不只是要求動作加快。
保留模型、振幅、獎勵、物理步長；不呼叫 learn，不保存或改寫模型。
無視窗為預設，--render 僅顯示 PPO。不要在評估期間拖曳模型或調整控制。
輸出 summary.csv、episodes.csv、每種頻率/控制器的逐步 CSV、metadata.json。
action_at_limit_fraction 計算施加 action 是否到邊界；不是力矩飽和比例。
rmse_after_warmup_deg 若沒有足夠資料則留空；failed_episodes 必須一起判讀。
這是單次訓練所得策略的敏感度測試，不是多個訓練 seed 的統計結論。
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
import experiment5_head_env as source


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(rows, warmup):
    errors = np.array([r["error_deg"] for r in rows])
    tail = np.array([r["error_deg"] for r in rows if r["time_s"] >= warmup])
    return dict(samples=len(rows), rmse_deg=float(np.sqrt(np.mean(errors**2))),
                rmse_after_warmup_deg=float(np.sqrt(np.mean(tail**2))) if len(tail) else None,
                max_abs_error_deg=float(np.max(np.abs(errors))),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                action_at_limit_fraction=float(np.mean([abs(r["action"]) >= .999999 for r in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/head_ppo_01"))
    p.add_argument("--frequencies", nargs="+", type=float, default=[.25, .5, .75, 1.])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--duration", type=float, default=10.)
    p.add_argument("--warmup", type=float, default=2.)
    p.add_argument("--eval-seed", type=int, default=10000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.episodes < 1 or args.eval_seed < 0:
        p.error("episodes >= 1 且 eval-seed >= 0")
    if not math.isfinite(args.duration) or not math.isfinite(args.warmup) or not 0 <= args.warmup < args.duration:
        p.error("必須 0 <= warmup < duration，且為有限數值")
    if any(not math.isfinite(f) or f <= 0 for f in args.frequencies):
        p.error("頻率必須為正有限數值")
    args.frequencies = list(dict.fromkeys(args.frequencies))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["env_sha256"] != sha256(source.__file__):
        raise RuntimeError("experiment5_head_env.py 與訓練時版本不同，請還原同一版本再測試")
    model_path = args.run / "final.zip"
    model_hash = sha256(model_path)
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    output = args.out or args.run / f"frequency_test_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(model_path=str(model_path.resolve()), model_sha256=model_hash,
                    training_frequency_hz=config["frequency"], frequencies_hz=args.frequencies,
                    duration_s=args.duration, warmup_s=args.warmup, episodes=args.episodes,
                    eval_seed=args.eval_seed, deterministic=True, env_sha256=config["env_sha256"],
                    versions={k: importlib.metadata.version(k) for k in
                              ["torch", "stable-baselines3", "mujoco", "gymnasium", "numpy"]})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summaries, episodes = [], []
    for index, frequency in enumerate(args.frequencies):
        for controller in ["direct", "feedback", "ppo"]:
            show = args.render and controller == "ppo"
            env = source.HeadTrackingEnv(frequency, args.duration, "human" if show else None)
            all_rows = []
            failures = 0
            viewer_used = False
            try:
                trace_path = output / f"{index:02d}_{frequency:g}Hz_{controller}.csv"
                with trace_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = None
                    for episode in range(args.episodes):
                        seed = args.eval_seed + episode
                        obs, _ = env.reset(seed=seed)
                        rows = []
                        for step in range(env.max_steps):
                            started = time.perf_counter()
                            if controller == "ppo":
                                # 唯一使用神經網路的地方；沒有 learn()。
                                action, _ = model.predict(obs, deterministic=True)
                            else:
                                action = source.baseline_action(obs, env, controller)
                            obs, reward, terminated, truncated, info = env.step(action)
                            row = dict(frequency_hz=frequency, controller=controller,
                                       episode=episode, seed=seed, step=step + 1, **info)
                            rows.append(row)
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            if show:
                                viewer_used = True
                                env.render()
                                if not env.viewer.is_running():
                                    raise KeyboardInterrupt("視窗已關閉，已寫入的逐步 CSV 保留")
                                time.sleep(max(0, env.dt - (time.perf_counter() - started)))
                            if terminated or truncated:
                                break
                        failures += int(terminated)
                        all_rows.extend(rows)
                        episodes.append(dict(frequency_hz=frequency, controller=controller,
                                             episode=episode, seed=seed, **metrics(rows, args.warmup),
                                             terminated=terminated, truncated=truncated))
                        stream.flush()
                result = dict(frequency_hz=frequency, controller=controller,
                              episodes=args.episodes, failed_episodes=failures,
                              **metrics(all_rows, args.warmup))
                summaries.append(result)
                write_rows(output / "summary.csv", summaries)
                write_rows(output / "episodes.csv", episodes)
                tail = result["rmse_after_warmup_deg"]
                tail_text = "N/A" if tail is None else f"{tail:.4f}"
                print(f"{frequency:g}Hz {controller:8s} RMSE={result['rmse_deg']:.4f} deg "
                      f"after_warmup={tail_text} reward={result['mean_reward']:.4f} "
                      f"failures={failures}", flush=True)
            finally:
                env.close()
                if viewer_used:
                    time.sleep(1.)
    if sha256(model_path) != model_hash:
        raise RuntimeError("模型檔案在評估期間被外部修改，請重新測試")
    print(f"模型 SHA256 未改變。Results: {output.resolve()}")


if __name__ == "__main__":
    main()
