#!/usr/bin/env python3
"""實驗三：head_yaw 正弦目標、追蹤誤差與身體姿態記錄。
使用 XML 位置致動器；沒有 RL、BAM 或主動平衡控制。
Python 3.10+ / mujoco==3.10.0 / numpy。
從 microduck_rl 根目錄執行，或用 --repo 指定其位置。
"""
import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

EXPERIMENT = 3


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("必須是有限數值")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="microduck_rl 專案根目錄")
    parser.add_argument("--scene", type=Path, help="直接指定場景 XML，優先於 --repo")
    parser.add_argument("--duration", type=finite_float, default=30.0,
                        help="模擬秒數，0 表示持續執行（預設 30）")
    parser.add_argument("--fix-base", action="store_true", help="以 weld 約束固定機身，隔離關節響應；不是平衡控制")
    parser.add_argument("--headless", action="store_true", help="不開視窗，以最快速度模擬")
    parser.add_argument("--csv", type=Path, help="輸出 CSV；預設使用時間戳檔名")
    if EXPERIMENT == 3:
        parser.add_argument("--amplitude-deg", type=finite_float, default=10.0)
        parser.add_argument("--frequency-hz", type=finite_float, default=0.25)
        parser.add_argument("--settle", type=finite_float, default=2.0,
                            help="正弦開始前維持 STAND 的模擬秒數")
        parser.add_argument("--ramp", type=finite_float, default=2.0,
                            help="振幅平滑升至設定值所需秒數（必須大於 0）")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration 不得小於 0")
    if EXPERIMENT == 3:
        if args.frequency_hz <= 0 or args.amplitude_deg < 0 or args.settle < 0 or args.ramp <= 0:
            parser.error("頻率、ramp 必須大於 0；振幅、settle 不得小於 0")
    return args


def find_scene(args):
    relative = Path("src/mjlab_microduck/robot/microduck/scene_walk.xml")
    if args.scene:
        candidates = [args.scene.expanduser()]
    elif args.repo:
        candidates = [args.repo.expanduser() / relative]
    else:
        here = Path(__file__).resolve().parent
        candidates = [Path.cwd() / relative, here / relative,
                      here.parent / "microduck_rl" / relative,
                      Path.cwd() / "microduck_rl" / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("找不到 scene_walk.xml。請使用 --repo /完整路徑/microduck_rl；需保留完整 assets 資料夾。")


def require_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"模型缺少必要名稱：{name}")
    return index


def run(args):
    scene = find_scene(args)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.005  # 200 Hz 物理步進；4 個物理步 = 1 次 50 Hz 命令更新
    data = mujoco.MjData(model)
    stand = require_id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    head = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_yaw")
    joint = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
    trunk = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    # actuator、joint、qpos、qvel 的索引不能假定相同，必須各自解析。
    q_index = model.jnt_qposadr[joint]
    v_index = model.jnt_dofadr[joint]
    mujoco.mj_resetDataKeyframe(model, data, stand)
    mujoco.mj_forward(model, data)
    if args.fix_base:
        # 把世界與機身用 weld 約束連接在 STAND 位置；不修改原始 XML。
        pose = [*data.xpos[trunk], *data.xquat[trunk]]
        spec = mujoco.MjSpec.from_file(str(scene))
        spec.add_equality(name="lesson_fixed_base", type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, name1="world", name2="trunk_base",
                          data=[0.0, 0.0, 0.0, *pose, 1.0], active=1)
        model = spec.compile()
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, stand)
        mujoco.mj_forward(model, data)
    home_ctrl = data.ctrl.copy()
    home_head = float(home_ctrl[head])
    if EXPERIMENT == 3:
        amplitude = math.radians(args.amplitude_deg)
        wanted = (home_head - amplitude, home_head + amplitude)
        for limited, limits, label in (
            (model.jnt_limited[joint], model.jnt_range[joint], "關節"),
            (model.actuator_ctrllimited[head], model.actuator_ctrlrange[head], "致動器"),
        ):
            if limited and (wanted[0] < limits[0] or wanted[1] > limits[1]):
                raise ValueError(f"要求的角度超出{label}範圍，請降低 --amplitude-deg")

    output = args.csv or Path(f"experiment{EXPERIMENT}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}.csv")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"MuJoCo {mujoco.__version__} | 場景：{scene}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu} | XML 位置致動器，無 RL / BAM")
    print(f"CSV：{output}")
    print("模式：固定機身 weld 約束" if args.fix_base else "模式：自由機身；固定 STAND 目標不保證平衡")
    print("關閉視窗或 Ctrl+C 結束。")
    print("視窗每 20 ms 更新；程式持續步進，viewer 的暫停按鈕不控制本程式。")
    fields = ["time_s", "command_time_s", "head_target_rad", "head_actual_rad",
              "head_error_rad", "head_velocity_rad_s", "trunk_x_m", "trunk_y_m",
              "trunk_z_m", "trunk_roll_deg", "trunk_pitch_deg", "trunk_yaw_deg",
              "trunk_tilt_deg", "contact_count"]
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or str(i)
        fields.append(f"ctrl_{name}_rad")
    if args.headless:
        view_context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer
        view_context = mj_viewer.launch_passive(model, data)
    steps = 0
    next_print = 0.0
    warned = False
    squared_error = 0.0
    samples = 0
    try:
        # x 模式避免意外覆蓋既有實驗紀錄。
        with output.open("x", newline="", encoding="utf-8") as file, view_context as viewer:
            writer = csv.writer(file)
            writer.writerow(fields)
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.xpos[trunk]
                    viewer.cam.distance = 0.65
                    viewer.cam.azimuth = 140
                    viewer.cam.elevation = -20
                viewer.sync()
            wall_start = time.perf_counter()
            while (viewer is None or viewer.is_running()) and (args.duration == 0 or data.time < args.duration - 1e-10):
                # 50 Hz 產生一次命令，在 4 個物理步中保持不變。
                command_time = float(data.time)
                data.ctrl[:] = home_ctrl
                if EXPERIMENT == 3:
                    elapsed = max(0.0, command_time - args.settle)
                    fraction = min(elapsed / args.ramp, 1.0)
                    envelope = 0.5 - 0.5 * math.cos(math.pi * fraction)
                    data.ctrl[head] = home_head + amplitude * envelope * math.sin(2 * math.pi * args.frequency_hz * elapsed)
                target = float(data.ctrl[head])
                for _ in range(4):
                    if args.duration > 0 and data.time >= args.duration - 1e-10:
                        break
                    mujoco.mj_step(model, data)
                    steps += 1
                    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                        raise RuntimeError("模擬出現非有限狀態，已停止；檢查參數與模型。")
                    if data.time + 1e-9 < steps * model.opt.timestep:
                        raise RuntimeError("MuJoCo 疑似因數值不穩定重設時間，已停止。")
                # 刷新 body transform，使 CSV 的姿態與當前 qpos 在同一時間點。
                mujoco.mj_forward(model, data)
                rotation = data.xmat[trunk].reshape(3, 3)
                roll = math.atan2(rotation[2, 1], rotation[2, 2])
                pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1, 1))))
                actual = float(data.qpos[q_index])
                error = target - actual
                writer.writerow([data.time, command_time, target, actual, error, data.qvel[v_index],
                                 *data.xpos[trunk], *map(math.degrees, (roll, pitch, yaw)), tilt,
                                 data.ncon, *data.ctrl])
                squared_error += error * error
                samples += 1
                if tilt > 45 and not warned:
                    print("警示：機身傾斜超過 45°；後續資料含失衡影響，不是孤立的頭部伺服測試。")
                    warned = True
                if data.time >= next_print:
                    print(f"t={data.time:6.2f}s  head target/actual={math.degrees(target):7.2f}/{math.degrees(actual):7.2f} deg  tilt={tilt:6.2f} deg")
                    next_print = float(data.time) + 1.0
                    file.flush()
                if viewer is not None:
                    viewer.sync()
                    # 絕對時間排程；過慢時不省略物理步，因此播放會慢於真實時間。
                    remaining = wall_start + float(data.time) - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n使用者中止，已保留 CSV。")
    if samples:
        rmse = math.degrees(math.sqrt(squared_error / samples))
        print(f"完成：{data.time:.3f} 模擬秒，{steps} 物理步，{samples} 筆紀錄。")
        print(f"全程頭部追蹤 RMSE={rmse:.3f} deg（包含初始暫態與失衡期間）。")
    print(f"紀錄：{output}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"錯誤：{exc}") from exc
