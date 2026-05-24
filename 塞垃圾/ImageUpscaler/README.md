# ImageUpscaler - 图片放大处理工具

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)](https://www.microsoft.com/windows)

一款功能强大的图片放大处理工具，支持多种常见图片格式和多种可自定义的放大方式，提供命令行和图形界面两种操作模式。

## ✨ 功能特性

- **支持多种图片格式**：JPG、JPEG、PNG、GIF、BMP、WEBP、TIFF
- **5种放大方式**：
  - 修改尺寸（直接调整像素大小）
  - 插值放大（最近邻、双线性、双三次、Lanczos）
  - AI智能放大（深度学习超分辨率）
  - 添加边框（保持原图，添加边框使整体变大）
  - 组合处理（依次应用所有方式）
- **批量处理**：支持单张或批量图片处理
- **实时进度**：处理进度条和预计剩余时间显示
- **详细日志**：完整的处理日志和报告生成
- **配置文件**：支持自定义默认参数

## 📦 下载与安装

### 方法一：直接下载EXE（推荐）

前往 [Releases](https://github.com/kuntang118/-/releases) 页面下载最新版本：

- `ImageUpscaler_GUI.exe` - 图形界面版本
- `ImageUpscaler_CLI.exe` - 命令行版本

无需安装Python，双击即可运行。

### 方法二：源码运行

```bash
# 克隆仓库
git clone https://github.com/kuntang118/-.git

# 安装依赖
pip install Pillow numpy opencv-python-headless

# 运行图形界面
python image_upscaler_gui.py

# 或运行命令行版本
python image_upscaler.py
```

## 🚀 快速开始

### 图形界面

1. 双击运行 `ImageUpscaler_GUI.exe`
2. 点击"添加文件"或"添加文件夹"导入图片
3. 选择放大方式和参数
4. 点击"开始处理"

### 命令行

```bash
# 放大单张图片2倍
ImageUpscaler_CLI.exe -i photo.jpg -s 2.0

# 使用双三次插值放大3倍
ImageUpscaler_CLI.exe -i photo.jpg -m interpolate -s 3.0 --interp bicubic

# 批量处理文件夹
ImageUpscaler_CLI.exe -i ./photos/ -m interpolate -s 2.0 -r

# 添加边框
ImageUpscaler_CLI.exe -i photo.jpg -m border -b 100

# 使用AI放大
ImageUpscaler_CLI.exe -i photo.jpg -m ai -s 2.0 --ai-model edsr
```

## 📖 详细文档

请参阅 [使用教程.md](使用教程.md) 获取完整的使用说明，包括：

- 图形界面详细教程
- 命令行参数完整说明
- 插值算法对比
- AI模型对比
- 常见问题解答

## 🛠️ 技术架构

```
ImageUpscaler/
├── image_upscaler.py          # 核心处理引擎（命令行版）
├── image_upscaler_gui.py      # 图形界面
├── upscaler_config.json       # 配置文件
└── 使用教程.md                # 详细使用教程
```

### 核心模块

- **ProcessorFactory**：处理器工厂，支持动态注册新算法
- **ResizeProcessor**：尺寸调整
- **InterpolateProcessor**：插值放大（OpenCV）
- **AIProcessor**：AI超分辨率（OpenCV DNN）
- **BorderProcessor**：边框填充
- **ProgressTracker**：进度跟踪

## 📋 系统要求

- **操作系统**：Windows 7/10/11
- **Python版本**：3.7+（源码运行）
- **依赖库**：
  - Pillow >= 8.0
  - NumPy >= 1.19
  - OpenCV-Python >= 4.5

## 🔧 插值算法对比

| 算法 | 速度 | 质量 | 特点 |
|------|------|------|------|
| 最近邻 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 速度快，边缘锯齿明显 |
| 双线性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 平衡速度和质量 |
| 双三次 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 质量较好 |
| Lanczos | ⭐⭐ | ⭐⭐⭐⭐⭐ | 质量最好，保留细节 |

## 🤖 AI模型对比

| 模型 | 速度 | 质量 | 特点 |
|------|------|------|------|
| EDSR | ⭐⭐ | ⭐⭐⭐⭐⭐ | 深度网络，细节最丰富 |
| ESPCN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 轻量级，适合实时处理 |
| FSRCNN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 快速超分辨率 |
| LapSRN | ⭐⭐⭐ | ⭐⭐⭐⭐ | 渐进式上采样 |

## 📝 更新日志

### v1.0 (2026-05-24)

- ✨ 初始版本发布
- 🎨 支持5种放大方式
- 💻 命令行和图形界面双模式
- 📦 支持批量处理
- ⚙️ 配置文件支持

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 👤 作者

- **kuntang118**
- GitHub: [@kuntang118](https://github.com/kuntang118)

---

如果这个项目对您有帮助，请给个 ⭐ Star 支持一下！
