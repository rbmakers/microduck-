#!/usr/bin/env python3
"""實驗 10：固定 action 延遲下，從頭訓練 PPO（五維觀測不變）。

與 experiment5_head_env.py、experiment9_action_delay.py 放同一資料夾。
使用實驗 6 虛擬環境，不需新增套件。此為單關節教學模型。

python experiment10_delay_training.py check
python experiment10_delay_training.py train --out runs/head_ppo_delay2_01
python experiment10_delay_training.py evaluate --run runs/head_ppo_delay2_01 --reference runs/head_ppo_01

固定 0.5 Hz / ±10 度，控制 50 Hz，物理 200 Hz。預設延遲 2 步。
延遲、previous_action 與 reward 直接沿用實驗 9 的 DelayedHeadEnv。
不加入 pending queue、不增加平滑懲罰、不更改物理參數。
這是固定延遲訓練，不是 domain randomization，不保證所有延遲都改善。
train 建立全新 PPO，並非從舊 final.zip 微調；預設與實驗 6 相同步數。
步數向上取整到 1024 的倍數；100000 實際為 100352。
evaluate 比較 direct、feedback、ppo_original、ppo_delay_trained。
所有控制器使用同一組 seeds；評估不 learn，不選擇最佳 checkpoint。
新目錄必須不存在，避免覆寫；舊模型只讀。config 記錄原始碼雜湊。
--render 僅顯示新策略；正式評估請用無視窗模式。
"""

import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import experiment5_head_env as source
import experiment9_action_delay as delay_source
from experiment9_action_delay import DelayedHeadEnv, sha256, metrics, write_rows


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def hashes():
    return dict(env_sha256=sha256(source.__file__),
                delay_env_sha256=sha256(delay_source.__file__),
                experiment10_sha256=sha256(__file__))


def versions():
    return {p: importlib.metadata.version(p) for p in
            ["stable-baselines3", "torch", "mujoco", "gymnasium", "numpy"]}


def verify_environment():
    """檢查真實環境：零延遲等價、FIFO/reset、觀測及 reward。"""
    base = source.HeadTrackingEnv()
    zero = DelayedHeadEnv(0)
    try:
        a, _ = base.reset(seed=123)
        b, _ = zero.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        for action in [.3, -.5, 1., -1., .2] * 5:
            a, ra, ta, xa, _ = base.step([action])
            b, rb, tb, xb, _ = zero.step([action])
            np.testing.assert_array_equal(a, b)
            assert (ra, ta, xa) == (rb, tb, xb)
    finally:
        base.close()
        zero.close()
    for d in [0, 1, 2, 4]:
        env = DelayedHeadEnv(d)
        try:
            check_env(env, warn=True)
            for seed in [123, 124]:
                obs, _ = env.reset(seed=seed)
                assert obs.shape == (5,) and obs[-1] == 0
                sent = [.2, -.4, .8, -1., 1., 0., -.1]
                for k, action in enumerate(sent):
                    obs, reward, _, _, info = env.step([action])
                    expected = sent[k-d] if k >= d else 0.
                    assert info['applied_action'] == expected
                    assert np.isclose(obs[-1], action)
                    previous = sent[k-1] if k else 0.
                    penalty = .02 * (action-previous)**2
                    assert np.isclose(info['smoothness_penalty'], penalty)
                    assert np.isclose(reward, info['tracking_reward']-penalty)
        finally:
            env.close()
    print("PASS: zero-delay equivalence, FIFO/reset, 5D observation, sent-action reward, SB3 check_env")


def train(args):
    verify_environment()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Monitor(DelayedHeadEnv(args.delay_steps), str(args.out / 'train.monitor.csv'))
    hyperparameters = dict(learning_rate=3e-4, n_steps=1024, batch_size=64,
                          n_epochs=10, gamma=.99, gae_lambda=.95, clip_range=.2,
                          ent_coef=0., vf_coef=.5, max_grad_norm=.5)
    config = dict(experiment=10, status='training', initialization='from_scratch',
                  frequency=.5, duration=10., control_dt_s=env.unwrapped.dt,
                  delay_steps=args.delay_steps, delay_ms=args.delay_steps*env.unwrapped.dt*1000,
                  seed=args.seed, requested_steps=args.steps, **hashes(), versions=versions(),
                  observation=['q', 'qdot', 'command', 'command_velocity', 'previous_sent_action'],
                  previous_action_semantics='last_sent', queue_initial_value=0.,
                  reward_semantics='tracking_minus_sent_action_difference_penalty',
                  hyperparameters=hyperparameters, network=dict(pi=[64,64], vf=[64,64]))
    save_json(args.out / 'config.json', config)
    try:
        # [A] 全新 actor/critic；與實驗 6 相同超參數與網路大小。
        model = PPO('MlpPolicy', env, **hyperparameters,
                    policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64])),
                    seed=args.seed, device='cpu', verbose=1)
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        model.save(args.out / 'initial')
        # [B] 唯一的 learn：rollout 已包含固定 action 延遲。
        model.learn(total_timesteps=args.steps,
                    callback=CheckpointCallback(save_freq=10000,
                                                save_path=str(args.out / 'checkpoints'),
                                                name_prefix='ppo_delay'))
        model.save(args.out / 'final')
        config.update(status='completed', actual_steps=model.num_timesteps,
                      model_sha256=sha256(args.out / 'final.zip'))
        save_json(args.out / 'config.json', config)
        print(f"Saved: {(args.out / 'final.zip').resolve()}")
    finally:
        env.close()


def load_config(run, delayed):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if config['env_sha256'] != sha256(source.__file__):
        raise RuntimeError(f'{run}: experiment5_head_env.py 與訓練時不同；請還原版本')
    if config['frequency'] != .5 or config['duration'] != 10.:
        raise RuntimeError('本比較限定 0.5 Hz、每回合 10 秒的訓練設定')
    if delayed:
        if config.get('experiment') != 10 or config.get('status') != 'completed':
            raise RuntimeError('新策略必須是完成訓練的實驗 10 模型')
        if config['delay_env_sha256'] != sha256(delay_source.__file__):
            raise RuntimeError('experiment9_action_delay.py 與訓練時版本不同')
        if config['model_sha256'] != sha256(run / 'final.zip'):
            raise RuntimeError('新模型 final.zip 與完成訓練時雜湊不同')
    elif config.get('delay_steps', 0) != 0:
        raise RuntimeError('reference 必須是原本無延遲訓練的模型')
    return config


def evaluate(args):
    configs = dict(ppo_original=load_config(args.reference, False),
                   ppo_delay_trained=load_config(args.run, True))
    paths = dict(ppo_original=args.reference / 'final.zip',
                 ppo_delay_trained=args.run / 'final.zip')
    if paths['ppo_original'].resolve() == paths['ppo_delay_trained'].resolve():
        raise ValueError('新舊模型必須是不同檔案')
    before = {k: sha256(v) for k,v in paths.items()}
    models = {k: PPO.load(v, device='cpu') for k,v in paths.items()}
    output = args.out or args.run / f'comparison_{time.time_ns()}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', model_paths={k:str(v.resolve()) for k,v in paths.items()},
                    model_sha256=before, training_configs=configs, **hashes(), versions=versions(),
                    delay_steps=args.delay_steps, frequency_hz=.5, duration_s=10.,
                    warmup_s=2., eval_seed=args.eval_seed, episodes=args.episodes,
                    deterministic=True, previous_action_semantics='last_sent')
    save_json(output / 'metadata.json', metadata)
    summaries, episodes = [], []
    for index, d in enumerate(args.delay_steps):
        for controller in ['direct', 'feedback', 'ppo_original', 'ppo_delay_trained']:
            show = args.render and controller == 'ppo_delay_trained'
            env = DelayedHeadEnv(d, render_mode='human' if show else None)
            all_rows, failures = [], 0
            try:
                with (output / f'{index:02d}_delay{d}steps_{controller}.csv').open(
                        'w', newline='', encoding='utf-8') as stream:
                    writer = None
                    for episode in range(args.episodes):
                        seed = args.eval_seed + episode
                        obs, _ = env.reset(seed=seed)
                        rows = []
                        for step in range(env.max_steps):
                            started = time.perf_counter()
                            # [C] 只 predict；所有控制器使用相同初始狀態。
                            action = (models[controller].predict(obs, deterministic=True)[0]
                                      if controller in models else source.baseline_action(obs, env, controller))
                            obs, _, terminated, truncated, info = env.step(action)
                            row = dict(delay_steps=d, delay_ms=d*env.dt*1000, frequency_hz=.5,
                                       controller=controller, episode=episode, seed=seed, step=step+1, **info)
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(row)
                            if show:
                                env.render()
                                if not env.viewer.is_running():
                                    raise KeyboardInterrupt('視窗已關閉；逐步資料保留，評估未完成')
                                time.sleep(max(0, env.dt-(time.perf_counter()-started)))
                            if terminated or truncated:
                                break
                        failures += int(terminated)
                        all_rows.extend(rows)
                        episodes.append(dict(delay_steps=d, delay_ms=d*env.dt*1000,
                                             frequency_hz=.5, controller=controller, episode=episode,
                                             seed=seed, **metrics(rows, 2.),
                                             terminated=terminated, truncated=truncated))
                        stream.flush()
                result = dict(delay_steps=d, delay_ms=d*env.dt*1000, frequency_hz=.5,
                              controller=controller, episodes=args.episodes,
                              failed_episodes=failures, **metrics(all_rows, 2.))
                summaries.append(result)
                write_rows(output / 'summary.csv', summaries)
                write_rows(output / 'episodes.csv', episodes)
                print(f"{d*env.dt*1000:g}ms {controller}: RMSE={result['rmse_deg']:.4f}, "
                      f"after2s={result['rmse_after_warmup_deg']}, failures={failures}", flush=True)
            finally:
                env.close()
                if show:
                    time.sleep(1.)
    after = {k:sha256(v) for k,v in paths.items()}
    if after != before:
        raise RuntimeError('評估期間模型被修改；本次結果不可使用')
    metadata.update(status='completed', models_unchanged=True)
    save_json(output / 'metadata.json', metadata)
    print(f'Results: {output.resolve()}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('check', help='不訓練，驗證實驗 9 延遲環境')
    tr = sub.add_parser('train')
    tr.add_argument('--out', type=Path, default=Path('runs/head_ppo_delay2_01'))
    tr.add_argument('--steps', type=int, default=100000)
    tr.add_argument('--delay-steps', type=int, default=2)
    tr.add_argument('--seed', type=int, default=42)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--run', type=Path, default=Path('runs/head_ppo_delay2_01'))
    ev.add_argument('--reference', type=Path, default=Path('runs/head_ppo_01'))
    ev.add_argument('--delay-steps', nargs='+', type=int, default=[0,1,2,4])
    ev.add_argument('--episodes', type=int, default=10)
    ev.add_argument('--eval-seed', type=int, default=10000)
    ev.add_argument('--out', type=Path)
    ev.add_argument('--render', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'train':
        if args.steps < 1 or args.delay_steps < 0 or not 0 <= args.seed < 2**32:
            p.error('steps > 0，delay-steps >= 0，seed 介於 0 與 2**32-1')
        train(args)
    elif args.mode == 'evaluate':
        if args.episodes < 1 or not 0 <= args.eval_seed < 2**32-args.episodes:
            p.error('episodes >= 1 且 eval-seed 必須在有效範圍')
        if any(d < 0 for d in args.delay_steps):
            p.error('delay-steps >= 0')
        args.delay_steps = list(dict.fromkeys(args.delay_steps))
        evaluate(args)
    else:
        verify_environment()


if __name__ == '__main__':
    main()
