#!/usr/bin/env python3
"""實驗 9：固定策略的 action 純延遲測試，不訓練。
與 experiment5_head_env.py 放同一資料夾，使用實驗 6 套件。
python experiment9_action_delay.py --run runs/head_ppo_01
python experiment9_action_delay.py --run runs/head_ppo_01 --delay-steps 4 --episodes 1 --render
預設 delay steps=0,1,2,4；dt=.02 秒；固定 .5 Hz / ±10 度 / 原始物理參數。
延遲 d 步：第 k 個控制區間施加 u[k-d]；負索引命令為 0。
這是控制命令傳送延遲，不是感測延遲，也不是改變致動器內部 PD 更新頻率。
obs 最後一項仍為上一個「送出」action；不提供延遲值或整個佇列給 PPO。
平滑懲罰針對相鄰「送出」action；另記錄 applied_smoothness_penalty。
CSV action/target_deg 保留「已施加」語意；sent_action/sent_target_deg 為新送出值。
零延遲時上述兩者相同。每次 reset 將佇列及歷史清零，避免跨回合殘留。
輸出 summary.csv、episodes.csv、各條件逐步 CSV、metadata.json。
--render 只顯示 PPO；保留一秒關窗緩衝。完整比較請使用無視窗模式。
這是測試單一已訓練模型，不是延遲補償訓練；未到失敗門檻不等於追蹤良好。
"""

from collections import deque
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



class DelayedHeadEnv(source.HeadTrackingEnv):
    """在原始環境前加 FIFO；保持政策觀測的 previous_action 語意。"""
    def __init__(self, delay_steps, frequency=.5, duration=10., render_mode=None):
        super().__init__(frequency, duration, render_mode)
        if not isinstance(delay_steps, int) or delay_steps < 0:
            raise ValueError("delay_steps 必須為非負整數")
        self.delay_steps = delay_steps
        self.pending = deque()
        self.previous_sent = 0.

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.pending = deque([0.] * self.delay_steps)
        self.previous_sent = 0.
        return obs, info

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("請先 reset")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError("action 必須為 shape=(1,) 的有限數值")
        sent = float(np.clip(a[0], -1, 1))
        # [A] 新命令排隊，取出最舊命令；d=0 時立即取出剛加入的值。
        self.pending.append(sent)
        applied = self.pending.popleft()
        obs, _, terminated, truncated, info = super().step(np.array([applied]))
        # [B] 策略知道自己上一步送出什麼，但看不到隱藏的佇列。
        obs[-1] = sent
        sent_penalty = self.smoothness_weight * (sent - self.previous_sent)**2
        info['applied_smoothness_penalty'] = info['smoothness_penalty']
        info['smoothness_penalty'] = sent_penalty
        reward = info['tracking_reward'] - sent_penalty
        info['reward'] = reward
        info['sent_action'] = sent
        info['sent_target_deg'] = math.degrees(self.action_scale * sent)
        info['applied_action'] = applied
        info['applied_target_deg'] = info['target_deg']
        self.previous_sent = sent
        return obs, float(reward), terminated, truncated, info


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
                action_at_limit_fraction=float(np.mean([abs(r["action"]) >= .999999 for r in rows])),
                sent_action_at_limit_fraction=float(np.mean([abs(r["sent_action"]) >= .999999 for r in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/head_ppo_01"))
    p.add_argument("--delay-steps", nargs="+", type=int, default=[0, 1, 2, 4])
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
    if any(d < 0 for d in args.delay_steps):
        p.error("delay-steps 必須為非負整數")
    args.delay_steps = list(dict.fromkeys(args.delay_steps))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["env_sha256"] != sha256(source.__file__):
        raise RuntimeError("experiment5_head_env.py 與訓練時版本不同，請還原同一版本再測試")
    model_path = args.run / "final.zip"
    model_hash = sha256(model_path)
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    output = args.out or args.run / f"delay_test_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(model_path=str(model_path.resolve()), model_sha256=model_hash,
                    training_frequency_hz=config["frequency"], test_frequency_hz=.5, delay_steps=args.delay_steps,
                    previous_action_semantics="last_sent", delay_queue_initial_value=0.,
                    smoothness_penalty_semantics="sent_action_difference",
                    duration_s=args.duration, warmup_s=args.warmup, episodes=args.episodes,
                    eval_seed=args.eval_seed, deterministic=True, env_sha256=config["env_sha256"],
                    versions={k: importlib.metadata.version(k) for k in
                              ["torch", "stable-baselines3", "mujoco", "gymnasium", "numpy"]})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summaries, episodes = [], []
    frequency = .5
    for index, delay_steps in enumerate(args.delay_steps):
        for controller in ["direct", "feedback", "ppo"]:
            show = args.render and controller == "ppo"
            env = DelayedHeadEnv(delay_steps, frequency, args.duration, "human" if show else None)
            delay_ms = delay_steps * env.dt * 1000
            metadata["control_dt_s"] = env.dt
            metadata["delay_ms"] = [d * env.dt * 1000 for d in args.delay_steps]
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            all_rows = []
            failures = 0
            viewer_used = False
            try:
                trace_path = output / f"{index:02d}_delay{delay_steps}steps_{controller}.csv"
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
                            row = dict(delay_steps=delay_steps, delay_ms=delay_ms, frequency_hz=frequency, controller=controller,
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
                        episodes.append(dict(delay_steps=delay_steps, delay_ms=delay_ms, frequency_hz=frequency, controller=controller,
                                             episode=episode, seed=seed, **metrics(rows, args.warmup),
                                             terminated=terminated, truncated=truncated))
                        stream.flush()
                result = dict(delay_steps=delay_steps, delay_ms=delay_ms, frequency_hz=frequency, controller=controller,
                              episodes=args.episodes, failed_episodes=failures,
                              **metrics(all_rows, args.warmup))
                summaries.append(result)
                write_rows(output / "summary.csv", summaries)
                write_rows(output / "episodes.csv", episodes)
                tail = result["rmse_after_warmup_deg"]
                tail_text = "N/A" if tail is None else f"{tail:.4f}"
                print(f"delay={delay_ms:g}ms {controller:8s} RMSE={result['rmse_deg']:.4f} deg "
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
