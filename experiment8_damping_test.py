#!/usr/bin/env python3
"""實驗 8：固定 PPO，測試關節被動黏滯阻尼的敏感度（不訓練）。
與 experiment5_head_env.py 放同一資料夾，使用實驗 6 的套件環境。
python experiment8_damping_test.py --run runs/head_ppo_01
python experiment8_damping_test.py --run runs/head_ppo_01 --damping-scales 2 --episodes 1 --render
預設阻尼倍率 0.5、1、2；固定命令 0.5 Hz、±10 度；每組 10 回合、每回合 10 秒。
只改 model.dof_damping[env.vid]，不改 XML 檔、不改策略、不改致動器 kp/kv。
注意 XML 關節 damping=.002，而位置致動器 kv=.02；這是不同來源。
每次建構環境均從原始值乘倍率，不會在回合之間累乘。
模型與環境來源雜湊都會核對；環境 reset 不會還原 model.dof_damping。
觀測不額外提供阻尼值，策略只能依原來五個觀測產生動作。
輸出 summary.csv、episodes.csv、逐步 CSV、parameters.csv、metadata.json。
這是固定參數測試，不是 domain randomization 訓練。
action_at_limit_fraction 是 action 邊界比例，不是力矩飽和比例。
無視窗為預設，--render 只顯示 PPO；保留一秒關閉緩衝。
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import mujoco
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
    p.add_argument("--damping-scales", nargs="+", type=float, default=[.5, 1., 2.])
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
    if any(not math.isfinite(f) or f <= 0 for f in args.damping_scales):
        p.error("阻尼倍率必須為正有限數值")
    args.damping_scales = list(dict.fromkeys(args.damping_scales))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    if config["env_sha256"] != sha256(source.__file__):
        raise RuntimeError("experiment5_head_env.py 與訓練時版本不同，請還原同一版本再測試")
    model_path = args.run / "final.zip"
    model_hash = sha256(model_path)
    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    output = args.out or args.run / f"damping_test_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(model_path=str(model_path.resolve()), model_sha256=model_hash,
                    training_frequency_hz=config["frequency"], test_frequency_hz=.5, damping_scales=args.damping_scales,
                    duration_s=args.duration, warmup_s=args.warmup, episodes=args.episodes,
                    eval_seed=args.eval_seed, deterministic=True, env_sha256=config["env_sha256"],
                    versions={k: importlib.metadata.version(k) for k in
                              ["torch", "stable-baselines3", "mujoco", "gymnasium", "numpy"]})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summaries, episodes, parameters = [], [], []
    frequency = .5
    for index, damping_scale in enumerate(args.damping_scales):
        for controller in ["direct", "feedback", "ppo"]:
            show = args.render and controller == "ppo"
            env = source.HeadTrackingEnv(frequency, args.duration, "human" if show else None)
            # [A] 只調整 head_yaw 對應自由度的被動阻尼。
            original_damping = float(env.model.dof_damping[env.vid])
            env.model.dof_damping[env.vid] = original_damping * damping_scale
            actual_damping = float(env.model.dof_damping[env.vid])
            # 此教學模型為 gear=1 的 position actuator；kv 編譯至 biasprm[2]。
            kp = float(env.model.actuator_gainprm[env.aid, 0])
            kv = -float(env.model.actuator_biasprm[env.aid, 2])
            parameters.append(dict(damping_scale=damping_scale, controller=controller,
                                   original_damping=original_damping, actual_damping=actual_damping,
                                   actuator_kp=kp, actuator_kv=kv,
                                   damping_plus_kv=actual_damping + kv))
            write_rows(output / "parameters.csv", parameters)
            all_rows = []
            failures = 0
            viewer_used = False
            try:
                trace_path = output / f"{index:02d}_damping{damping_scale:g}x_{controller}.csv"
                with trace_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = None
                    for episode in range(args.episodes):
                        seed = args.eval_seed + episode
                        obs, _ = env.reset(seed=seed)
                        if not np.isclose(env.model.dof_damping[env.vid], actual_damping):
                            raise RuntimeError("reset 改變阻尼設定，停止測試")
                        mujoco.mj_forward(env.model, env.data)
                        rows = []
                        for step in range(env.max_steps):
                            started = time.perf_counter()
                            if controller == "ppo":
                                # 唯一使用神經網路的地方；沒有 learn()。
                                action, _ = model.predict(obs, deterministic=True)
                            else:
                                action = source.baseline_action(obs, env, controller)
                            obs, reward, terminated, truncated, info = env.step(action)
                            row = dict(damping_scale=damping_scale, damping=actual_damping, frequency_hz=frequency, controller=controller,
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
                        episodes.append(dict(damping_scale=damping_scale, damping=actual_damping, frequency_hz=frequency, controller=controller,
                                             episode=episode, seed=seed, **metrics(rows, args.warmup),
                                             terminated=terminated, truncated=truncated))
                        stream.flush()
                result = dict(damping_scale=damping_scale, damping=actual_damping, frequency_hz=frequency, controller=controller,
                              episodes=args.episodes, failed_episodes=failures,
                              **metrics(all_rows, args.warmup))
                summaries.append(result)
                write_rows(output / "summary.csv", summaries)
                write_rows(output / "episodes.csv", episodes)
                tail = result["rmse_after_warmup_deg"]
                tail_text = "N/A" if tail is None else f"{tail:.4f}"
                print(f"damping={damping_scale:g}x ({actual_damping:g}) {controller:8s} RMSE={result['rmse_deg']:.4f} deg "
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
