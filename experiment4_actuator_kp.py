#!/usr/bin/env python3
"""實驗四：讀取有效參數，只修改 head_yaw 的 Kp。

快速開始（把本檔放在 microduck_rl 根目錄）：
  python experiment4_actuator_kp.py --inspect-only
  python experiment4_actuator_kp.py --kp-scale 0.5
  python experiment4_actuator_kp.py --kp-scale 1
  python experiment4_actuator_kp.py --kp-scale 2

預設固定機身、±10 度、0.5 Hz、30 秒；--free-base 可解除固定。
每次建立獨立 CSV 與同名 .parameters.json，保留完整實驗配置。
--headless 不開視窗；macOS 開視窗時使用 mjpython 代替 python。

閱讀順序：parse_args → find_scene → run 中的 A/B/C/D/E 區段。
重點：Kp 必須同時寫入 gainprm[head,0] 和 biasprm[head,1] 的負值。
只改 gain 而不改 bias，會改變目標與實際角度的關係，並非單純改 Kp。
Kv、關節 damping/frictionloss/armature、其他致動器均保持原值。

CSV：time_s 是區間末時間，command_time_s 是命令產生時間。
每 20 ms 一列；head_torque_nm 是區間末的重新求解值。
interval_peak_abs_torque_nm 是該區間 4 個物理步起點的最大絕對力矩；
interval_force_limit_fraction 是其中未限幅需求達到輸出界限的比例。
這些是離散取樣，不代表連續時間的絕對最大值。
穩態 RMSE 使用 command_time_s >= settle+ramp；不補償相位或 20 ms 時戳差。

本程式只接受無 activation dynamics、gear=1、直接 hinge 傳動的
固定增益/仿射偏置 position 型態，避免把其他致動器誤當位置控制器。
XML 不修改。預設模型會簡化部分碰撞，--free-base 不是完整倒地測試。
參考模型：microduck_rl commit 53b8971b61baf5b7f3c16d135dd7cac37623de4b。

使用 XML 位置致動器；沒有 RL、BAM 或主動平衡控制。
Python 3.10+ / mujoco==3.10.0 / numpy。
從 microduck_rl 根目錄執行，或用 --repo 指定其位置。
"""
import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

EXPERIMENT = 4


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
    parser.add_argument("--free-base", dest="fix_base", action="store_false",
                        help="解除預設的固定機身 weld 約束")
    parser.set_defaults(fix_base=True)
    parser.add_argument("--inspect-only", action="store_true", help="只列印參數，不開視窗或模擬")
    parser.add_argument("--kp-scale", type=finite_float, default=1.0,
                        help="head_yaw Kp 相對 XML 原值的倍數（預設 1）")
    parser.add_argument("--headless", action="store_true", help="不開視窗，以最快速度模擬")
    parser.add_argument("--csv", type=Path, help="輸出 CSV；預設使用時間戳檔名")
    if EXPERIMENT == 4:
        parser.add_argument("--amplitude-deg", type=finite_float, default=10.0)
        parser.add_argument("--frequency-hz", type=finite_float, default=0.5)
        parser.add_argument("--settle", type=finite_float, default=2.0,
                            help="正弦開始前維持 STAND 的模擬秒數")
        parser.add_argument("--ramp", type=finite_float, default=2.0,
                            help="振幅平滑升至設定值所需秒數（必須大於 0）")
    args = parser.parse_args()
    if args.kp_scale <= 0:
        parser.error("--kp-scale 必須大於 0")
    if args.duration < 0:
        parser.error("--duration 不得小於 0")
    if EXPERIMENT == 4:
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


def validate_position(model, head, joint):
    """本教學的 Nm / rad 解讀限於此種直接傳動。"""
    kp = float(model.actuator_gainprm[head, 0])
    valid = (
        model.actuator_trntype[head] == mujoco.mjtTrn.mjTRN_JOINT
        and model.actuator_trnid[head, 0] == joint
        and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
        and np.allclose(model.actuator_gear[head], [1, 0, 0, 0, 0, 0])
        and model.actuator_dyntype[head] == mujoco.mjtDyn.mjDYN_NONE
        and model.actuator_gaintype[head] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[head] == mujoco.mjtBias.mjBIAS_AFFINE
        and kp > 0
        and np.isclose(model.actuator_biasprm[head, 0], 0)
        and np.isclose(model.actuator_biasprm[head, 1], -kp)
    )
    if not valid:
        raise ValueError("head_yaw 不是本教學支援的 gear=1、無動態位置伺服，拒絕修改。")


def inspect_parameters(model, head, joint):
    dof = int(model.jnt_dofadr[joint])
    return {
        "mujoco_version": mujoco.__version__,
        "actuator_name": "head_yaw", "actuator_id": int(head),
        "joint_id": int(joint), "qpos_index": int(model.jnt_qposadr[joint]),
        "qvel_index": dof, "nu": int(model.nu),
        "kp_original": float(model.actuator_gainprm[head, 0]),
        "kv_effective": float(-model.actuator_biasprm[head, 2]),
        "gainprm": model.actuator_gainprm[head].tolist(),
        "biasprm": model.actuator_biasprm[head].tolist(),
        "gear": model.actuator_gear[head].tolist(),
        "control_limited": bool(model.actuator_ctrllimited[head]),
        "control_range_rad": model.actuator_ctrlrange[head].tolist(),
        "force_limited": bool(model.actuator_forcelimited[head]),
        "force_range_nm": model.actuator_forcerange[head].tolist(),
        "joint_damping": float(model.dof_damping[dof]),
        "joint_frictionloss": float(model.dof_frictionloss[dof]),
        "joint_armature": float(model.dof_armature[dof]),
        "joint_limited": bool(model.jnt_limited[joint]),
        "joint_range_rad": model.jnt_range[joint].tolist(),
    }


def torque_diagnostics(model, data, head):
    # 使用 actuator length/velocity 重建一般化伺服，避免忽略 joint reference。
    u = float(data.ctrl[head])
    if model.actuator_ctrllimited[head]:
        u = float(np.clip(u, *model.actuator_ctrlrange[head]))
    gain = float(model.actuator_gainprm[head, 0])
    b0, b1, b2 = model.actuator_biasprm[head, :3]
    raw = float(gain*u + b0 + b1*data.actuator_length[head] + b2*data.actuator_velocity[head])
    limited = False
    if model.actuator_forcelimited[head]:
        lo, hi = model.actuator_forcerange[head]
        limited = raw <= lo or raw >= hi
    return raw, float(data.actuator_force[head]), int(limited)


def run(args):
    # A. 載入 XML，建立狀態、找出各種索引。
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
    # B. 檢查有效參數；不要把任意 general/motor 當成 position。
    validate_position(model, head, joint)
    parameters = inspect_parameters(model, head, joint)
    kp_original = parameters["kp_original"]
    kp_used = kp_original * args.kp_scale
    print("\n=== XML 編譯後的有效參數 ===")
    print(json.dumps(parameters, ensure_ascii=False, indent=2))
    print(f"\nhead_yaw: Kp {kp_original:g} × {args.kp_scale:g} = {kp_used:g}")

    # C. 只在記憶體中修改頭部 Kp。位置伺服：p = kp*u - kp*length - kv*velocity。
    model.actuator_gainprm[head, 0] = kp_used
    model.actuator_biasprm[head, 1] = -kp_used
    mujoco.mj_forward(model, data)
    if args.inspect_only:
        print("僅檢查模式：未執行物理步進，也未修改 XML。")
        return
    home_ctrl = data.ctrl.copy()
    home_head = float(home_ctrl[head])
    if EXPERIMENT == 4:
        amplitude = math.radians(args.amplitude_deg)
        wanted = (home_head - amplitude, home_head + amplitude)
        for limited, limits, label in (
            (model.jnt_limited[joint], model.jnt_range[joint], "關節"),
            (model.actuator_ctrllimited[head], model.actuator_ctrlrange[head], "致動器"),
        ):
            if limited and (wanted[0] < limits[0] or wanted[1] > limits[1]):
                raise ValueError(f"要求的角度超出{label}範圍，請降低 --amplitude-deg")

    output = args.csv or Path(f"experiment{EXPERIMENT}_kp{args.kp_scale:g}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}.csv")
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
    parameters.update({"scene": str(scene), "kp_scale": args.kp_scale,
                       "kp_used": kp_used, "gainprm_used": model.actuator_gainprm[head].tolist(),
                       "biasprm_used": model.actuator_biasprm[head].tolist(),
                       "fixed_base": args.fix_base, "frequency_hz": args.frequency_hz,
                       "amplitude_deg": args.amplitude_deg, "duration_s": args.duration,
                       "settle_s": args.settle, "ramp_s": args.ramp,
                       "physics_dt_s": model.opt.timestep, "control_dt_s": 0.02,
                       "integrator": int(model.opt.integrator)})
    metadata_path = output.with_suffix(".parameters.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("CSV 或 parameters.json 已存在，請改用新檔名。")
    with metadata_path.open("x", encoding="utf-8") as meta_file:
        json.dump(parameters, meta_file, ensure_ascii=False, indent=2)
    fields += ["kp_scale", "kp_used", "kv_used", "head_torque_raw_nm", "head_torque_nm",
               "interval_peak_abs_torque_nm", "interval_force_limit_fraction"]
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
    steady_squared = 0.0
    steady_samples = 0
    steady_limit_hits = 0
    steady_physics_samples = 0
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
                # D. 50 Hz 產生一次角度命令（控制器仍在 MuJoCo 內），在 4 個物理步中保持不變。
                command_time = float(data.time)
                data.ctrl[:] = home_ctrl
                if EXPERIMENT == 4:
                    elapsed = max(0.0, command_time - args.settle)
                    fraction = min(elapsed / args.ramp, 1.0)
                    envelope = 0.5 - 0.5 * math.cos(math.pi * fraction)
                    data.ctrl[head] = home_head + amplitude * envelope * math.sin(2 * math.pi * args.frequency_hz * elapsed)
                target = float(data.ctrl[head])
                interval_peak = 0.0
                interval_hits = 0
                interval_samples = 0
                # E. 在每個物理步起點記錄力矩，再推進物理。
                for _ in range(4):
                    if args.duration > 0 and data.time >= args.duration - 1e-10:
                        break
                    mujoco.mj_forward(model, data)
                    raw, force, hit = torque_diagnostics(model, data, head)
                    interval_peak = max(interval_peak, abs(force))
                    interval_hits += hit
                    interval_samples += 1
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
                raw, force, _ = torque_diagnostics(model, data, head)
                writer.writerow([data.time, command_time, target, actual, error, data.qvel[v_index],
                                 *data.xpos[trunk], *map(math.degrees, (roll, pitch, yaw)), tilt,
                                 data.ncon, *data.ctrl, args.kp_scale, kp_used,
                                 parameters["kv_effective"], raw, force,
                                 interval_peak, interval_hits / interval_samples])
                if command_time >= args.settle + args.ramp - 1e-9:
                    steady_squared += error * error
                    steady_samples += 1
                    steady_limit_hits += interval_hits
                    steady_physics_samples += interval_samples
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
    if steady_samples:
        print(f"漸增段後 RMSE={math.degrees(math.sqrt(steady_squared / steady_samples)):.3f} deg")
        print(f"漸增段後取樣達力矩界限比例={100*steady_limit_hits/steady_physics_samples:.2f}%")
    print(f"紀錄：{output}\n參數：{metadata_path}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f"錯誤：{exc}") from exc
