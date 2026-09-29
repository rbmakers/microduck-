#!/usr/bin/env python3
"""實驗 5：MuJoCo → Gymnasium 單關節追蹤環境（尚未訓練 PPO）。

安裝：python -m pip install "mujoco>=3.2,<4" "gymnasium>=1.0,<2" numpy
驗證：python experiment5_head_env.py --check
執行：python experiment5_head_env.py --controller direct --render
比較：python experiment5_head_env.py --controller zero
      python experiment5_head_env.py --controller feedback
macOS 視窗模式請改用 mjpython experiment5_head_env.py --render

教學用模型：固定底座、單一 head_yaw；尺寸、質量、增益均為示例，
不是 microduck 官方模型／BAM 馬達模型，不載入 scene_walk.xml。
模型在下方 XML 字串，以 MjModel.from_xml_string(XML) 載入。

action=[a]，a ∈ [-1,1]，致動器目標 = a * 20 度（內部 rad）。
observation=[q, qdot, command, command_velocity, previous_action]。
前四項單位依序 rad, rad/s, rad, rad/s；最後一項無單位。
command 與 command_velocity 一起描述固定頻率正弦命令的相位。
reset 回傳 (obs, info)，step 回傳 (obs, reward, terminated, truncated, info)。
本環境不自動 reset；episode 結束後必須由呼叫端 reset。
每步 reward 是步末追蹤分數減動作變化懲罰，沒有再乘 dt。
CSV 以步末時間對齊 command/q；action 是剛才這個區間使用的命令。
沒有 action 訓練或網路更新；三種控制器只是環境測試基準。

閱讀順序：XML → __init__ → reset → step → baseline_action → main。
"""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


# [A] 簡化模型就在這裡。base 沒有 joint，所以固定於世界座標。
XML = """
<mujoco model="teaching_head_yaw">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 -1 2"/>
    <geom type="plane" size="1 1 .1" rgba=".8 .8 .8 1"/>
    <body name="base" pos="0 0 .15">
      <geom type="box" size=".06 .05 .10" rgba=".2 .3 .5 1"/>
      <body name="head" pos="0 0 .14">
        <joint name="head_yaw" type="hinge" axis="0 0 1"
               limited="true" range="-.6 .6" damping=".002" armature=".001"/>
        <geom type="box" size=".06 .04 .035" mass=".10" rgba=".9 .7 .1 1"/>
        <geom type="box" pos=".07 0 0" size=".025 .025 .012"
              mass=".01" rgba="1 .3 .1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="head_servo" joint="head_yaw" kp=".25" kv=".02"
              ctrllimited="true" ctrlrange="-.3490658504 .3490658504"
              forcelimited="true" forcerange="-.08 .08"/>
  </actuator>
</mujoco>
"""


class HeadTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, frequency=0.5, duration=10.0, render_mode=None):
        super().__init__()
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency 必須為正有限值")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration 必須為正有限值")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode 必須為 None 或 human")
        # [B] 載入 XML；本實驗完全不依賴外部 mesh/XML。
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
        self.qid = int(self.model.jnt_qposadr[jid])
        self.vid = int(self.model.jnt_dofadr[jid])
        self.aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_servo")
        self.decimation = 4
        self.dt = float(self.model.opt.timestep) * self.decimation
        self.max_steps = math.ceil(duration / self.dt)
        self.frequency = frequency
        self.amplitude = math.radians(10)
        self.action_scale = math.radians(20)
        self.error_scale = math.radians(5)
        self.smoothness_weight = 0.02
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # 無限界限讓 checker 接受合理瞬態；不是觀測正規化。
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None
        self.needs_reset = True

    def command(self):
        phase = 2 * math.pi * self.frequency * float(self.data.time) + self.phase0
        c = self.amplitude * math.sin(phase)
        cdot = self.amplitude * 2 * math.pi * self.frequency * math.cos(phase)
        return c, cdot

    def _obs(self):
        c, cdot = self.command()
        return np.array([self.data.qpos[self.qid], self.data.qvel[self.vid],
                         c, cdot, self.previous_action], dtype=np.float32)

    # [C] 重設物理狀態、歷史、計數器及命令相位。
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.phase0 = float(self.np_random.uniform(-math.pi, math.pi))
        self.data.qpos[self.qid] = self.np_random.uniform(-math.radians(2), math.radians(2))
        self.previous_action = 0.0
        self.steps = 0
        self.needs_reset = False
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"time_s": 0.0}

    # [D] RL 一步 = action 轉換 + 四次物理步 + 評分 + 觀測。
    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("請先 reset；episode 結束後也需要 reset")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("action 必須是含一個有限數值的陣列，shape=(1,)")
        a = float(np.clip(action[0], -1, 1))
        target = self.action_scale * a
        self.data.ctrl[self.aid] = target
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            self.needs_reset = True
            raise FloatingPointError("模擬出現非有限值，請檢查物理設定")
        self.steps += 1
        c, _ = self.command()  # 步末命令，與步末實際角度比較
        q = float(self.data.qpos[self.qid])
        error = q - c
        tracking = math.exp(-(error / self.error_scale) ** 2)
        penalty = self.smoothness_weight * (a - self.previous_action) ** 2
        reward = tracking - penalty
        # 教學用失敗門檻：不是 microduck 跌倒判斷。
        terminated = bool(abs(q) > math.radians(30))
        truncated = bool(self.steps >= self.max_steps)
        self.previous_action = a
        self.needs_reset = terminated or truncated
        info = dict(time_s=float(self.data.time), command_deg=math.degrees(c),
                    target_deg=math.degrees(target), actual_deg=math.degrees(q),
                    error_deg=math.degrees(error), action=a,
                    tracking_reward=tracking, smoothness_penalty=penalty,
                    reward=reward, terminated=terminated, truncated=truncated)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, .22]
                self.viewer.cam.distance = .8
                self.viewer.cam.elevation = -25
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            print("[close] requesting viewer shutdown", flush=True)

            self.viewer.close()

            # 暫時保留 Python 程序與 model/data，
            # 讓背景視窗有時間完成退出與資源清理。
            time.sleep(1.0)

            self.viewer = None
            print("[close] shutdown grace period finished", flush=True)


def baseline_action(obs, env, mode):
    """手寫控制器，不是訓練後的 policy；feedback 係數只是示例。"""
    q, qdot, command, command_velocity, _ = map(float, obs)
    if mode == "zero":
        target = 0.0
    elif mode == "direct":
        target = command
    else:
        target = command + 0.35 * (command - q) + 0.025 * (command_velocity - qdot)
    return np.array([np.clip(target / env.action_scale, -1, 1)], dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=["zero", "direct", "feedback"], default="direct")
    p.add_argument("--frequency", type=float, default=.5)
    p.add_argument("--duration", type=float, default=10.)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes 必須至少為 1")
    env = HeadTrackingEnv(args.frequency, args.duration, "human" if args.render else None)
    try:
        if args.check:
            from gymnasium.utils.env_checker import check_env
            check_env(env, skip_render_check=True)
            print("Gymnasium check_env: PASS（未檢查視窗）")
            return
        path = args.output or Path(f"experiment5_{args.controller}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["episode", "step", "time_s", "command_deg", "target_deg", "actual_deg",
                  "error_deg", "action", "tracking_reward", "smoothness_penalty", "reward",
                  "terminated", "truncated"]
        with path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for episode in range(args.episodes):
                # 相同 seed/episode 在不同控制器之間提供相同初始條件。
                obs, _ = env.reset(seed=args.seed + episode)
                errors, rewards = [], []
                for step in range(env.max_steps):
                    started = time.perf_counter()
                    action = baseline_action(obs, env, args.controller)
                    obs, reward, terminated, truncated, info = env.step(action)
                    writer.writerow(dict(episode=episode, step=step + 1, **info))
                    errors.append(info["error_deg"])
                    rewards.append(reward)
                    if args.render:
                        env.render()
                        if not env.viewer.is_running():
                            print(f"視窗關閉；部分記錄已寫入 {path.resolve()}")
                            return
                        time.sleep(max(0, env.dt - (time.perf_counter() - started)))
                    if terminated or truncated:
                        break
                print(f"episode={episode} steps={len(errors)} RMSE={np.sqrt(np.mean(np.square(errors))):.3f} deg "
                      f"mean_reward={np.mean(rewards):.4f} terminated={terminated} truncated={truncated}")
        print(f"CSV: {path.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
