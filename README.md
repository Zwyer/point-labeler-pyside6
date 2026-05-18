# 点云语义标注工具（PySide6 + PyVista）

一个面向 SemanticKITTI 数据格式的桌面点云标注工具，基于 PySide6 构建界面，使用 PyVista/VTK 进行 3D 渲染与交互。

## 功能特性

- 读取 SemanticKITTI 单序列点云：
  - `sequence_dir/velodyne/*.bin`
  - 自动匹配/创建 `sequence_dir/labels/*.label`
- 点云显示与交互：
  - 点大小调节
  - 黑夜/白天主题切换
  - 视角重置
- 标注模式：
  - 浏览模式
  - 框选模式（多边形框选）
  - 刷子模式（按屏幕像素半径刷选）
- 类别管理：
  - 新增/删除类别
  - 修改类别颜色
  - 类别锁定（防止误改）
  - 类别显隐（隐藏点位）
- 标签编辑能力：
  - 批量修改语义标签
  - 撤销（Undo）
  - 自动保存（切帧/切序列时）
- 类别配置导入导出：
  - `json`
  - `yaml/yml`（需安装 `pyyaml`）

## 环境要求

- Python 3.10+（建议使用 3.10~3.13）
- Windows（当前脚本示例以 Windows 为主）

## 安装依赖

```powershell
python -m pip install numpy PySide6 pyvista pyvistaqt vtk
# 如需导入/导出 YAML 类别配置
python -m pip install pyyaml
```

## 快速启动

方式一：直接运行批处理脚本

```powershell
.\run.bat
```

方式二：命令行运行

```powershell
$env:PYTHONPATH="src"
python -m point_labeler.app.main
```

## 数据目录结构

选择序列目录时，支持以下形式：

- `.../sequences/00`
- `.../sequences/00/velodyne`

推荐目录结构：

```text
00/
  velodyne/
    000000.bin
    000001.bin
    ...
  labels/
    000000.label
    000001.label
    ...
```

说明：

- `*.bin` 为 `float32` 的 `N x 4`（`x, y, z, intensity`）
- `*.label` 为 `uint32` 的一维数组
- 语义/实例编码遵循 SemanticKITTI：
  - 低 16 位：`semantic_id`
  - 高 16 位：`instance_id`

## 类别配置文件格式

JSON / YAML 结构一致，均为数组，每项包含 `id/name/color`：

```json
[
  { "id": 0, "name": "未标注", "color": "#808080" },
  { "id": 1, "name": "汽车", "color": "#e74c3c" },
  { "id": 2, "name": "行人", "color": "#2ecc71" }
]
```

## 交互说明

- `Alt + 鼠标`：进入标注交互（非浏览模式）
- 刷子模式：
  - `Alt + 左键拖动`：连续刷选
  - `Alt + 滚轮`：调节刷子半径
- 框选模式：
  - `Alt + 左键`：添加/拖动顶点
  - `Alt + 右键` 或 `Alt + 双击左键`：应用框选
  - `Alt + 中键`：清空当前多边形
- `Ctrl + Z`：撤销

## 打包为可执行文件（可选）

项目已提供 `PointLabeler.spec` 与构建脚本：

```powershell
python -m pip install pyinstaller
.\build.bat
```

构建产物：

- `dist/PointLabeler/PointLabeler.exe`

## 当前实现说明

- 工具当前主要编辑语义标签（`semantic_id`）
- 实例标签（`instance_id`）会按当前逻辑保留或置零后写回

## 许可证

建议在公开仓库中使用 MIT 许可证（宽松、易于复用）。  
如你已选择其他许可证，请以仓库中的 `LICENSE` 文件为准。

