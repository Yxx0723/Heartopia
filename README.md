# 心动小镇视觉自动资源采集程序

这是一个 Windows PC 端的视觉自动资源采集 MVP。程序只使用游戏窗口截图、
OpenCV/模板匹配和普通键鼠输入，视觉、状态机、路线和输入控制彼此解耦。

## 已完成范围

Task 1–20 已完成：

- Windows 游戏窗口查找、客户区截图和 BGR 帧接口；
- Debug Overlay、FPS 和状态显示；
- 默认 dry-run 的键鼠输入控制器，以及输入异常时的统一释放；
- `ResourceDetector`、`MockResourceDetector` 和单资源 OpenCV 模板匹配；
- 目标选择、屏幕中心比例转向、短步接近和交互提示模板检测；
- 基于 tick 的固定路线，可在发现目标时暂停并在采集后恢复；
- MotionDetector 与固定步骤 Recovery；
- BotEngine、SafetyGuard、Loguru 日志、失败截图和 JSON；
- F6 截图、F8 暂停、F9 恢复、F10 停止；
- CLI、启动脚本、示例配置和自动化测试。
- 启动前配置诊断和 `--check` 检查；
- 模板质量检查、ROI 裁剪保存和静态截图校准工具；
- 背包满、异常 UI、窗口前台状态检测；
- 20 目标离线状态机压测和目标丢失/Recovery/UI 故障注入；
- `pyproject.toml`、发布验收脚本和 Windows 发布清单。

程序明确不读取或修改游戏内存，不注入 DLL，不 Hook 游戏进程，不分析或修改
网络封包，不实现反作弊绕过、反检测随机化、隐藏进程、验证码绕过或自动登录。
自动化行为可能违反游戏服务条款，使用前请确认目标环境允许此类自动化。

## 安装

建议使用 Python 3.11 或更高版本，并在 Windows 虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

YOLO 仅保留为后续可选扩展，不是当前 MVP 的运行依赖：

```powershell
python -m pip install -r requirements-yolo.txt
```

## 启动

在 `heartopia_bot` 目录执行：

```powershell
# Debug：显示 Overlay，并强制 dry-run，不发送真实游戏输入
python main.py --debug

# 正常启动（默认配置仍是 dry-run，改配置前请确认安全边界）
python main.py --config config/default.yaml

# 录制画面到 dataset/images/，默认按 1 秒间隔保存
python main.py --record

# 覆盖固定路线
python main.py --route config/routes/test_route.yaml

# 只检查配置，不查找游戏窗口、不启动输入
python main.py --check --config config/default.yaml

# 离线模拟 20 个目标和故障路径
python -m tools.soak_test --targets 20

# 发布前完整检查
python -m tools.release_check
```

也可以双击：

- `start_debug.bat`：Debug + dry-run；
- `start_normal.bat`：使用默认配置。
- `start_record.bat`：定时录制画面；
- `check_config.bat`：运行配置诊断；
- `run_soak_test.bat`：运行 20 目标离线压测。

首次实际运行前，需要让游戏窗口可见，并确保标题包含
`config/default.yaml` 中的 `window.title_contains`。默认 detector 是 `mock`，
用于先验证状态机和输入安全；使用 OpenCV 资源识别时，把 `vision.detector` 改为
`opencv`，并配置模板路径。

配置诊断会检查窗口标题、FPS、路线文件、模板文件、热键冲突、输出目录和真实
输入警告。诊断不查找窗口，也不会发送键鼠输入。

### 制作模板

先用 F6 保存一张截图，然后通过静态截图框选模板：

```powershell
python -m tools.calibrate_template `
  --input dataset/images/screenshot_xxx.png `
  --output assets/resources/wood.png
```

也可以直接指定客户区像素框：

```powershell
python -m tools.calibrate_template `
  --input dataset/images/screenshot_xxx.png `
  --output assets/resources/wood.png `
  --bbox 800 400 960 600
```

## 默认快捷键

快捷键在配置的热键轮询中按下沿触发：

- F6：保存当前帧到 `dataset/images/`；
- F8：暂停状态机、路线并释放所有按键；
- F9：恢复暂停前状态；
- F10：立即释放输入并停止主循环。

快捷键配置示例：

```yaml
hotkeys:
  screenshot: F6
  pause: F8
  resume: F9
  stop: F10
```

## 配置重点

完整示例见 [config/default.yaml](config/default.yaml)。
路线由动作组成，支持 `move_forward`、`move_backward`、`strafe_left`、
`strafe_right`、`turn_left`、`turn_right` 和 `wait`。移动动作使用
`duration`，转向动作使用 `mouse_dx`；也兼容规格中的旧字段 `pixels`。

默认安全配置：

```yaml
control:
  dry_run: true

safety:
  pause_when_unfocused: true
  max_runtime_minutes: 60
```

只有目标窗口在前台且未暂停、未超时，SafetyGuard 才允许向输入控制器发送
动作。真实输入模式需要用户明确把 `control.dry_run` 改为 `false`，并自行确认
风险。

## 测试

```powershell
pytest -q
```

测试覆盖配置、窗口和截图替身、Overlay、输入安全、Mock 和 OpenCV 检测、目标
选择、转向、状态机完整路径、路线中断恢复、卡死恢复、安全保护、引擎、日志相关
截图、快捷键边沿检测、配置诊断、模板工具、UI 阻断和 20 目标离线压测。

发布前检查的详细步骤见 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)。

## 已知限制

- 当前 MVP 只针对一种资源类型进行识别；
- `mock` detector 是默认值，真实 OpenCV detector 需要用户准备模板图；
- 背包满和异常 UI 当前通过模板检测，需要用户提供对应模板；
- `--record` 以固定时间间隔采集截图，不自动生成标注；
- 离线 20 目标压测只验证状态机和故障路径，不代表真实游戏现场结果；
- 没有实现 YOLO 训练、世界坐标导航、小地图规划、自动回家/售卖/背包整理等
  非 MVP 功能；
- 是否能稳定识别和交互取决于游戏 UI、分辨率、窗口缩放和模板质量，必须先用
  dry-run 在允许的测试环境中验证。
