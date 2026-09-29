# microduck 學習實驗：從 MuJoCo 到 PPO（Zero-to-Policy Labs）

以 [microduck](https://github.com/pollen-robotics/microduck_rl) 開源專案為藍本的教學實驗。前半用 MuJoCo 理解機器人物理模擬與位置致動器，後半把單一關節 `head_yaw` 寫成 Gymnasium 環境，用 PPO 訓練追蹤策略，並測試策略對頻率、阻尼、延遲的敏感度。



> 這是教學用簡化模型，不是 microduck 官方模型、BAM 馬達模型或官方 mjlab／rsl_rl 訓練流程。

## 實驗一覽

| 實驗 | 檔案 | 主題 | 需要 microduck_rl |
|---|---|---|---|
| 2 | `experiment2_stand.py` | STAND keyframe、自由／固定機身模擬迴圈 | 是 |
| 3 | `experiment3_head_sine.py` | head_yaw 正弦命令、追蹤誤差記錄 | 是 |
| 4 | `experiment4_actuator_kp.py` | 只改 Kp，理解 position 致動器 | 是 |
| 5 | `experiment5_head_env.py` | 單關節 Gymnasium 環境與三種手寫基準 | 否（內嵌 XML） |
| 6 | `experiment6_ppo.py` | Stable-Baselines3 PPO 訓練與評估 | 否 |
| 7 | `experiment7_frequency_test.py` | 固定策略的頻率泛化測試 | 否 |
| 8 | `experiment8_damping_test.py` | 關節黏滯阻尼敏感度 | 否 |
| 9 | `experiment9_action_delay.py` | action 純延遲測試 | 否 |

詳細講義（導論與各單元原理、程式結構、結果判讀）見 [`microduck_lecture.html`](microduck_lecture.html)。

## 共同設計

- 物理步長 0.005 s（200 Hz），4 個物理步為 1 次控制（dt = 0.02 s，50 Hz）。
- 實驗 5–9 的觀測為 `[q, q̇, command, command_velocity, previous_action]`，動作為 `a ∈ [-1, 1]`，目標角度 = `a × 20°`。
- Reward：`exp(-(e/5°)²) − 0.02·(a − a_prev)²`。
- 評估使用固定 seed（10000 起），三種手寫控制器（zero／direct／feedback）與 PPO 看到完全相同的命令。
- 實驗 6–9 會核對 `experiment5_head_env.py` 與模型檔的 SHA256，確保比較的是同一環境與同一模型。

## 環境需求

Python ≥ 3.10。

```bash
# 實驗 2–4（版本與參考專案 lockfile 一致）
python -m pip install "mujoco==3.10.0" numpy

# 實驗 5
python -m pip install "mujoco>=3.2,<4" "gymnasium>=1.0,<2" numpy

# 實驗 6–9
python -m pip install "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "stable-baselines3==2.9.0"
```

實驗 2–4 需要 microduck_rl（參考 commit `53b8971`）及其完整 `assets/` 資料夾，可放在根目錄執行或以 `--repo /path/to/microduck_rl` 指定。macOS 開視窗請用 `mjpython` 取代 `python`。

## 快速開始

```bash
# 實驗 2–4：固定機身頭部測試
python experiment4_actuator_kp.py --inspect-only
python experiment4_actuator_kp.py --kp-scale 0.5 --headless

# 實驗 5：驗證環境並跑基準控制器
python experiment5_head_env.py --check
python experiment5_head_env.py --controller feedback

# 實驗 6：訓練與評估（每次請使用新的輸出資料夾）
python experiment6_ppo.py train --steps 100000 --out runs/head_ppo_01
python experiment6_ppo.py evaluate --run runs/head_ppo_01 --episodes 10

# 實驗 7–9：不訓練，只測固定模型
python experiment7_frequency_test.py --run runs/head_ppo_01
python experiment8_damping_test.py   --run runs/head_ppo_01
python experiment9_action_delay.py   --run runs/head_ppo_01
```

## 主要結果

單一訓練 seed（42），10 回合，命令 ±10°，暫態（前 2 s）後 RMSE。

**訓練效果（0.5 Hz）**

| 控制器 | RMSE (°) |
|---|---|
| zero | 7.071 |
| direct | 2.175 |
| feedback | 1.594 |
| PPO initial（未訓練） | 7.051 |
| PPO final | 0.047 |

zero 基準等於 10°/√2 = 7.07°，可作為指標正確性的檢查點；initial ≈ zero 表示改善來自學習。

**頻率泛化（訓練頻率 0.5 Hz）**

| 頻率 | direct | feedback | PPO |
|---|---|---|---|
| 0.25 Hz | 1.068 | 0.788 | 0.022 |
| 0.5 Hz | 2.175 | 1.594 | 0.047 |
| 0.75 Hz | 3.353 | 2.432 | 0.095 |
| 1.0 Hz | 4.610 | 3.308 | 0.169 |

**阻尼敏感度**：阻尼放大 4 倍（0.5×→2×），手寫控制器誤差僅增約 12%。致動器 kv（0.02）遠大於關節 damping（0.002），所以此測試對模型的擾動有限。

**Action 延遲（0.5 Hz）**

| 延遲 | direct | feedback | PPO | PPO action 觸限比例 |
|---|---|---|---|---|
| 0 ms | 2.175 | 1.594 | 0.047 | 0.6% |
| 20 ms | 2.616 | 1.920 | 2.509 | 60% |
| 40 ms | 3.053 | 2.246 | 6.601 | 81% |
| 80 ms | 3.919 | 2.904 | 12.092 | 88% |

- 手寫控制器的延遲退化符合相位模型：direct 在 80 ms 的預測 RMSE 3.92°，實測 3.919°。
- PPO 在零延遲下的高精度依賴高有效增益；只加 20 ms 延遲就進入飽和，80 ms 時比什麼都不做（7.07°）更差。這是 sim-to-real 的典型風險，對策包括訓練時隨機化延遲、在觀測中加入 action 歷史，或提高平滑懲罰。
- 「延遲造成高增益振盪」是合理推論，尚未用逐步 CSV 驗證。

## 限制

- 只有單一訓練 seed，不是統計結論。
- 頻率、阻尼、延遲為「訓練後單獨擾動」的固定參數測試，不是 domain randomization 訓練。
- 終止門檻（|q| > 30°）是教學設定，未失敗不代表追蹤良好。
- `action_at_limit_fraction` 是 action 邊界比例，不是力矩飽和比例。
- 模型尺寸、質量與增益為示例值，不代表 microduck 硬體。

## 與 microduck 的對應

| 本專案 | microduck |
|---|---|
| 1 維 action、5 維 observation | 14 維 action、61 維 observation |
| SB3 PPO，單環境，MLP 64×64 | rsl_rl PPO，4096 環境，MLP 512→256→128 |
| XML position 致動器 | BAM 馬達模型 |
| 測試後單獨擾動 | 訓練期 domain randomization |
| 教學 reward | 速度追蹤、直立、腳部離地時間、動作變化等多項 |

## 授權與致謝

請在此補上本儲存庫的授權條款。microduck 專案與其模型版權屬原作者（Pollen Robotics）；本儲存庫的程式為獨立的教學實驗，使用 microduck_rl 的場景檔做為實驗 2–4 的輸入。
