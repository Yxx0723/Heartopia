# 发布验收清单

## 自动检查

在项目根目录执行：

```powershell
python main.py --check --config config/default.yaml
python -m tools.soak_test --targets 20
python -m tools.release_check
```

`tools.release_check` 会检查关键文件、默认配置、Python 编译和 pytest。

## Windows 现场检查

1. 安装 `requirements-dev.txt`。
2. 启动游戏并确认窗口标题包含配置值。
3. 运行 `start_debug.bat`，保持 `control.dry_run: true`。
4. 确认窗口截图和 Overlay 正常，测试 F8/F9/F10 能释放和停止。
5. 准备模板后切换 OpenCV detector，先运行 `--check`。
6. 在允许自动化的测试环境中做至少 20 个目标的现场 dry-run。
7. 只有确认安全和权限后，才考虑关闭 dry-run。

## 发布边界

本项目没有自动登录、内存读取、DLL 注入、进程 Hook、封包分析、驱动输入、
反作弊绕过或反检测随机化。发布前不得通过配置或外部脚本绕过这些边界。
