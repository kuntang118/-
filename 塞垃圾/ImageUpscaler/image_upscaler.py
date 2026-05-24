# -*- coding: utf-8 -*-
"""
图片放大处理工具 v1.0
功能：支持多种图片格式的放大处理，提供多种放大方式

支持的放大方式：
  1. resize   - 修改图片尺寸（直接调整像素大小）
  2. interpolate - 图像插值放大（最近邻、双线性、双三次、Lanczos）
  3. ai       - AI内容感知放大（使用深度学习超分辨率）
  4. border   - 添加边框/填充（保持原图内容，添加边框使整体变大）
  5. all      - 组合使用以上所有方式

支持的图片格式：JPG、JPEG、PNG、GIF、BMP、WEBP、TIFF

使用方式：
  python image_upscaler.py                          # 交互模式
  python image_upscaler.py -i input.jpg -s 2.0      # 放大2倍
  python image_upscaler.py -i input.jpg -m resize -s 2.0
  python image_upscaler.py -i folder/ -m all -s 2.0  # 批量处理
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


# ============================================================
# 配置和常量
# ============================================================

SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif')
DEFAULT_OUTPUT_DIR = 'output_upscaled'
LOG_DIR = 'logs'
CONFIG_FILE = 'upscaler_config.json'


class InterpolationMethod(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class UpscaleMethod(Enum):
    RESIZE = "resize"
    INTERPOLATE = "interpolate"
    AI = "ai"
    BORDER = "border"
    ALL = "all"


@dataclass
class ProcessingConfig:
    """处理配置类"""
    scale_factor: float = 2.0
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    interpolation: InterpolationMethod = InterpolationMethod.LANCZOS
    border_size: int = 50
    border_color: Tuple[int, int, int] = (255, 255, 255)
    output_format: str = "png"
    quality: int = 95
    output_dir: str = DEFAULT_OUTPUT_DIR
    ai_model: str = "edsr"
    ai_scale: int = 2
    keep_aspect_ratio: bool = True
    sharpen: bool = False
    denoise: bool = False

    def to_dict(self):
        return {
            'scale_factor': self.scale_factor,
            'target_width': self.target_width,
            'target_height': self.target_height,
            'interpolation': self.interpolation.value,
            'border_size': self.border_size,
            'border_color': self.border_color,
            'output_format': self.output_format,
            'quality': self.quality,
            'output_dir': self.output_dir,
            'ai_model': self.ai_model,
            'ai_scale': self.ai_scale,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'sharpen': self.sharpen,
            'denoise': self.denoise,
        }

    @classmethod
    def from_dict(cls, data: dict):
        if 'interpolation' in data:
            data['interpolation'] = InterpolationMethod(data['interpolation'])
        return cls(**data)


# ============================================================
# 日志设置
# ============================================================

def setup_logging():
    """设置日志记录"""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'upscaler_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


# ============================================================
# 配置文件管理
# ============================================================

def load_config() -> ProcessingConfig:
    """加载配置文件"""
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ProcessingConfig.from_dict(data)
        except Exception as e:
            logging.warning(f"加载配置文件失败: {e}，使用默认配置")
    return ProcessingConfig()


def save_config(config: ProcessingConfig):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
        logging.info(f"配置已保存到 {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"保存配置文件失败: {e}")


# ============================================================
# 进度显示
# ============================================================

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total: int, desc: str = "处理中"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, increment: int = 1):
        """更新进度"""
        with self.lock:
            self.current += increment
            self._print_progress()
    
    def _print_progress(self):
        """打印进度条"""
        if self.total == 0:
            return
        
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"预计剩余: {eta:.1f}秒"
        else:
            eta_str = "预计剩余: 计算中..."
        
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '#' * filled + '-' * (bar_length - filled)
        
        print(f"\r  {self.desc}: [{bar}] {percentage:.1f}% ({self.current}/{self.total}) {eta_str}", end='', flush=True)
        
        if self.current >= self.total:
            print()


# ============================================================
# 图片处理核心模块
# ============================================================

class ImageProcessor:
    """图片处理器基类"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def process(self, image: Image.Image) -> Image.Image:
        """处理图片，子类必须实现"""
        raise NotImplementedError
    
    def get_output_suffix(self) -> str:
        """获取输出文件后缀标识"""
        raise NotImplementedError


class ResizeProcessor(ImageProcessor):
    """尺寸调整处理器"""
    
    def process(self, image: Image.Image) -> Image.Image:
        """修改图片尺寸"""
        config = self.config
        
        if config.target_width and config.target_height:
            new_size = (config.target_width, config.target_height)
        else:
            scale = config.scale_factor
            new_size = (
                int(image.width * scale),
                int(image.height * scale)
            )
        
        # 选择插值方法
        interp_map = {
            InterpolationMethod.NEAREST: Image.Resampling.NEAREST,
            InterpolationMethod.BILINEAR: Image.Resampling.BILINEAR,
            InterpolationMethod.BICUBIC: Image.Resampling.BICUBIC,
            InterpolationMethod.LANCZOS: Image.Resampling.LANCZOS,
        }
        interp = interp_map.get(config.interpolation, Image.Resampling.LANCZOS)
        
        result = image.resize(new_size, interp)
        
        # 后处理
        if config.sharpen:
            result = result.filter(ImageFilter.SHARPEN)
        
        return result
    
    def get_output_suffix(self) -> str:
        return f"_resize_{self.config.scale_factor}x"


class InterpolateProcessor(ImageProcessor):
    """插值放大处理器"""
    
    def process(self, image: Image.Image) -> Image.Image:
        """使用插值算法放大图片"""
        config = self.config
        
        # 转换为OpenCV格式进行处理
        img_array = np.array(image)
        
        if config.target_width and config.target_height:
            new_size = (config.target_width, config.target_height)
        else:
            scale = config.scale_factor
            new_size = (
                int(image.width * scale),
                int(image.height * scale)
            )
        
        # OpenCV插值方法映射
        cv_interp_map = {
            InterpolationMethod.NEAREST: cv2.INTER_NEAREST,
            InterpolationMethod.BILINEAR: cv2.INTER_LINEAR,
            InterpolationMethod.BICUBIC: cv2.INTER_CUBIC,
            InterpolationMethod.LANCZOS: cv2.INTER_LANCZOS4,
        }
        cv_interp = cv_interp_map.get(config.interpolation, cv2.INTER_LANCZOS4)
        
        # 处理不同通道数
        if len(img_array.shape) == 2:  # 灰度图
            result = cv2.resize(img_array, new_size, interpolation=cv_interp)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            # 分别处理RGB和Alpha通道
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
            
            rgb_resized = cv2.resize(rgb, new_size, interpolation=cv_interp)
            alpha_resized = cv2.resize(alpha, new_size, interpolation=cv2.INTER_LINEAR)
            
            result = np.dstack((rgb_resized, alpha_resized))
        else:  # RGB
            result = cv2.resize(img_array, new_size, interpolation=cv_interp)
        
        # 后处理
        if config.denoise:
            if len(result.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
            else:
                result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
        
        if config.sharpen:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(result, -1, kernel)
        
        return Image.fromarray(result)
    
    def get_output_suffix(self) -> str:
        return f"_interp_{self.config.interpolation.value}_{self.config.scale_factor}x"


class AIProcessor(ImageProcessor):
    """AI内容感知放大处理器"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载AI超分辨率模型"""
        try:
            # 使用OpenCV的DNN超分辨率模块
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            
            model_name = self.config.ai_model
            scale = self.config.ai_scale
            
            # 模型文件路径（需要预先下载）
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            
            model_files = {
                'edsr': f'EDSR_x{scale}.pb',
                'espcn': f'ESPCN_x{scale}.pb',
                'fsrcnn': f'FSRCNN_x{scale}.pb',
                'lapsrn': f'LapSRN_x{scale}.pb',
            }
            
            model_file = model_dir / model_files.get(model_name, f'EDSR_x{scale}.pb')
            
            if not model_file.exists():
                logging.warning(f"AI模型文件不存在: {model_file}")
                logging.info("将使用传统插值方法作为替代")
                self.sr = None
                return
            
            self.sr.readModel(str(model_file))
            self.sr.setModel(model_name, scale)
            logging.info(f"AI模型加载成功: {model_name} x{scale}")
            
        except Exception as e:
            logging.warning(f"AI模型加载失败: {e}")
            logging.info("将使用传统插值方法作为替代")
            self.sr = None
    
    def process(self, image: Image.Image) -> Image.Image:
        """使用AI算法放大图片"""
        if self.sr is None:
            logging.info("AI模型不可用，回退到Lanczos插值")
            fallback = InterpolateProcessor(self.config)
            return fallback.process(image)
        
        img_array = np.array(image)
        
        # OpenCV DNN超分辨率只支持3通道BGR
        if len(img_array.shape) == 2:
            # 灰度图转BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            result = self.sr.upsample(img_bgr)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA图：分别处理RGB和Alpha
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
            
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr_upscaled = self.sr.upsample(bgr)
            rgb_upscaled = cv2.cvtColor(bgr_upscaled, cv2.COLOR_BGR2RGB)
            
            # Alpha通道使用双线性插值
            alpha_upscaled = cv2.resize(alpha, (rgb_upscaled.shape[1], rgb_upscaled.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)
            
            result = np.dstack((rgb_upscaled, alpha_upscaled))
        else:
            # RGB图
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result_bgr = self.sr.upsample(bgr)
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result)
    
    def get_output_suffix(self) -> str:
        return f"_ai_{self.config.ai_model}_{self.config.ai_scale}x"


class BorderProcessor(ImageProcessor):
    """边框填充处理器"""
    
    def process(self, image: Image.Image) -> Image.Image:
        """添加边框使整体尺寸变大"""
        config = self.config
        border = config.border_size
        color = config.border_color
        
        # 确保颜色通道数匹配
        if image.mode == 'RGBA':
            if len(color) == 3:
                color = (*color, 255)
        elif image.mode == 'L':
            if len(color) == 3:
                color = (int(sum(color) / 3),)
        
        result = ImageOps.expand(image, border=border, fill=color)
        return result
    
    def get_output_suffix(self) -> str:
        return f"_border_{self.config.border_size}px"


class CombinedProcessor(ImageProcessor):
    """组合处理器：依次应用所有方法"""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.processors = [
            ResizeProcessor(config),
            InterpolateProcessor(config),
            AIProcessor(config),
            BorderProcessor(config),
        ]
    
    def process(self, image: Image.Image) -> Image.Image:
        """依次应用所有处理方法"""
        result = image
        for processor in self.processors:
            try:
                result = processor.process(result)
            except Exception as e:
                logging.warning(f"处理器 {processor.__class__.__name__} 失败: {e}")
        return result
    
    def get_output_suffix(self) -> str:
        return f"_combined_{self.config.scale_factor}x"


# ============================================================
# 处理器工厂
# ============================================================

class ProcessorFactory:
    """处理器工厂"""
    
    _processors = {
        UpscaleMethod.RESIZE: ResizeProcessor,
        UpscaleMethod.INTERPOLATE: InterpolateProcessor,
        UpscaleMethod.AI: AIProcessor,
        UpscaleMethod.BORDER: BorderProcessor,
        UpscaleMethod.ALL: CombinedProcessor,
    }
    
    @classmethod
    def create(cls, method: UpscaleMethod, config: ProcessingConfig) -> ImageProcessor:
        """创建处理器实例"""
        processor_class = cls._processors.get(method)
        if processor_class is None:
            raise ValueError(f"未知的处理方法: {method}")
        return processor_class(config)
    
    @classmethod
    def register(cls, method: str, processor_class: type):
        """注册新的处理器"""
        cls._processors[UpscaleMethod(method)] = processor_class


# ============================================================
# 图片加载和保存
# ============================================================

def load_image(path: str) -> Image.Image:
    """加载图片文件"""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式: {file_path.suffix}")
    
    try:
        image = Image.open(file_path)
        
        # 处理GIF动画（取第一帧）
        if image.format == 'GIF' and getattr(image, 'is_animated', False):
            image.seek(0)
        
        # 确保图片数据已加载
        image.load()
        
        return image
    except Exception as e:
        raise RuntimeError(f"加载图片失败 {path}: {e}")


def save_image(image: Image.Image, path: str, format: str = None, quality: int = 95):
    """保存图片文件"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format is None:
        format = output_path.suffix.lstrip('.').upper()
    
    # 统一格式名称
    format_upper = format.upper()
    if format_upper == 'JPG':
        format_upper = 'JPEG'
    
    save_kwargs = {}
    
    if format_upper == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
        # JPEG不支持透明通道
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
    elif format_upper == 'PNG':
        save_kwargs['optimize'] = True
    elif format_upper == 'WEBP':
        save_kwargs['quality'] = quality
    elif format_upper == 'GIF':
        save_kwargs['optimize'] = True
    
    try:
        image.save(output_path, format=format_upper, **save_kwargs)
        return output_path
    except Exception as e:
        raise RuntimeError(f"保存图片失败 {path}: {e}")


# ============================================================
# 批量处理引擎
# ============================================================

def process_single_image(
    input_path: str,
    processor: ImageProcessor,
    config: ProcessingConfig,
    output_dir: Path
) -> Dict:
    """处理单张图片"""
    result = {
        'input': input_path,
        'status': 'pending',
        'message': '',
        'output': None,
        'original_size': None,
        'final_size': None,
        'processing_time': 0,
    }
    
    try:
        start_time = time.time()
        
        # 加载图片
        image = load_image(input_path)
        result['original_size'] = (image.width, image.height)
        
        # 处理图片
        processed = processor.process(image)
        result['final_size'] = (processed.width, processed.height)
        
        # 生成输出文件名
        input_path_obj = Path(input_path)
        suffix = processor.get_output_suffix()
        output_name = f"{input_path_obj.stem}{suffix}.{config.output_format}"
        output_path = output_dir / output_name
        
        # 保存图片
        save_image(processed, str(output_path), config.output_format, config.quality)
        
        result['processing_time'] = time.time() - start_time
        result['status'] = 'success'
        result['output'] = str(output_path)
        result['message'] = f"成功: {input_path_obj.name} -> {output_name}"
        
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"错误: {str(e)}"
        logging.error(f"处理失败 {input_path}: {e}")
    
    return result


def process_batch(
    input_paths: List[str],
    method: UpscaleMethod,
    config: ProcessingConfig,
    progress_callback: Callable = None
) -> List[Dict]:
    """批量处理图片"""
    results = []
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建处理器
    processor = ProcessorFactory.create(method, config)
    
    # 进度跟踪
    progress = ProgressTracker(len(input_paths), f"处理图片 ({method.value})")
    
    for i, input_path in enumerate(input_paths):
        result = process_single_image(input_path, processor, config, output_dir)
        results.append(result)
        
        # 更新进度
        progress.update()
        
        if progress_callback:
            progress_callback(i + 1, len(input_paths), result)
    
    return results


# ============================================================
# 文件扫描
# ============================================================

def scan_images(path: str, recursive: bool = False) -> List[str]:
    """扫描图片文件"""
    target = Path(path)
    images = []
    
    if target.is_file():
        if target.suffix.lower() in SUPPORTED_FORMATS:
            images.append(str(target))
    elif target.is_dir():
        pattern = '**/*' if recursive else '*'
        for file_path in target.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                images.append(str(file_path))
    
    return sorted(images)


# ============================================================
# 结果报告
# ============================================================

def generate_report(results: List[Dict], method: UpscaleMethod, config: ProcessingConfig) -> str:
    """生成处理报告"""
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    report_lines = [
        "=" * 70,
        "                    图片放大处理报告",
        "=" * 70,
        f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"处理方法: {method.value}",
        f"放大倍数: {config.scale_factor}x",
        f"插值算法: {config.interpolation.value}",
        f"输出格式: {config.output_format}",
        f"输出质量: {config.quality}",
        "-" * 70,
        f"总计处理: {len(results)} 张图片",
        f"成功: {success_count} 张",
        f"失败: {error_count} 张",
        "-" * 70,
    ]
    
    for result in results:
        status_icon = "[OK]" if result['status'] == 'success' else "[ERR]"
        report_lines.append(f"  {status_icon} {result['message']}")
        
        if result['status'] == 'success':
            orig = result.get('original_size')
            final = result.get('final_size')
            proc_time = result.get('processing_time', 0)
            if orig and final:
                report_lines.append(f"      尺寸: {orig[0]}x{orig[1]} -> {final[0]}x{final[1]}")
            report_lines.append(f"      耗时: {proc_time:.2f}秒")
    
    report_lines.extend([
        "-" * 70,
        f"输出目录: {Path(config.output_dir).absolute()}",
        "=" * 70,
    ])
    
    return '\n'.join(report_lines)


def save_report(report: str):
    """保存报告到文件"""
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_file


# ============================================================
# 交互式界面
# ============================================================

def interactive_mode():
    """交互式命令行界面"""
    print("=" * 70)
    print("              图片放大处理工具 v1.0 - 交互模式")
    print("=" * 70)
    
    config = load_config()
    
    # 选择输入
    print("\n【步骤1】选择输入")
    input_path = input("  请输入图片文件或文件夹路径: ").strip()
    
    if not input_path or not Path(input_path).exists():
        print("  [ERR] 路径无效")
        return
    
    # 扫描图片
    recursive = input("  是否递归扫描子文件夹? (y/N): ").strip().lower() == 'y'
    images = scan_images(input_path, recursive)
    
    if not images:
        print("  [ERR] 未找到支持的图片文件")
        return
    
    print(f"  找到 {len(images)} 张图片")
    for img in images[:5]:
        print(f"    - {Path(img).name}")
    if len(images) > 5:
        print(f"    ... 还有 {len(images) - 5} 张")
    
    # 选择处理方法
    print("\n【步骤2】选择放大方式")
    print("  1. resize    - 修改图片尺寸")
    print("  2. interpolate - 图像插值放大")
    print("  3. ai        - AI内容感知放大")
    print("  4. border    - 添加边框/填充")
    print("  5. all       - 组合所有方式")
    
    method_choice = input("  请选择 (1-5，默认2): ").strip()
    method_map = {
        '1': UpscaleMethod.RESIZE,
        '2': UpscaleMethod.INTERPOLATE,
        '3': UpscaleMethod.AI,
        '4': UpscaleMethod.BORDER,
        '5': UpscaleMethod.ALL,
    }
    method = method_map.get(method_choice, UpscaleMethod.INTERPOLATE)
    
    # 设置参数
    print("\n【步骤3】设置参数")
    
    scale_input = input(f"  放大倍数 (默认{config.scale_factor}): ").strip()
    if scale_input:
        config.scale_factor = float(scale_input)
    
    if method in (UpscaleMethod.INTERPOLATE, UpscaleMethod.ALL):
        print("  插值算法: 1.最近邻 2.双线性 3.双三次 4.Lanczos")
        interp_choice = input("  请选择 (1-4，默认4): ").strip()
        interp_map = {
            '1': InterpolationMethod.NEAREST,
            '2': InterpolationMethod.BILINEAR,
            '3': InterpolationMethod.BICUBIC,
            '4': InterpolationMethod.LANCZOS,
        }
        config.interpolation = interp_map.get(interp_choice, InterpolationMethod.LANCZOS)
    
    if method in (UpscaleMethod.BORDER, UpscaleMethod.ALL):
        border_input = input(f"  边框大小 (默认{config.border_size}px): ").strip()
        if border_input:
            config.border_size = int(border_input)
    
    output_format = input(f"  输出格式 (png/jpg/webp，默认{config.output_format}): ").strip()
    if output_format:
        config.output_format = output_format
    
    quality_input = input(f"  输出质量 1-100 (默认{config.quality}): ").strip()
    if quality_input:
        config.quality = int(quality_input)
    
    # 确认处理
    print("\n【确认】")
    print(f"  处理方法: {method.value}")
    print(f"  放大倍数: {config.scale_factor}x")
    print(f"  输出格式: {config.output_format}")
    print(f"  处理数量: {len(images)} 张")
    
    confirm = input("  确认开始处理? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("  已取消")
        return
    
    # 开始处理
    print("\n" + "=" * 70)
    results = process_batch(images, method, config)
    print("=" * 70)
    
    # 生成报告
    report = generate_report(results, method, config)
    print("\n" + report)
    
    # 保存报告
    report_file = save_report(report)
    print(f"\n报告已保存: {report_file}")
    
    # 保存配置
    save_config(config)


# ============================================================
# 命令行参数解析
# ============================================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='图片放大处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -i image.jpg -s 2.0                    # 放大2倍
  %(prog)s -i image.jpg -m interpolate -s 3.0     # 使用插值放大3倍
  %(prog)s -i folder/ -m all -s 2.0 -f png        # 批量处理，输出PNG
  %(prog)s -i image.jpg -m border -b 100          # 添加100px边框
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入图片文件或文件夹路径')
    parser.add_argument('-m', '--method', default='interpolate',
                       choices=['resize', 'interpolate', 'ai', 'border', 'all'],
                       help='放大方式 (默认: interpolate)')
    parser.add_argument('-s', '--scale', type=float, default=2.0,
                       help='放大倍数 (默认: 2.0)')
    parser.add_argument('--width', type=int,
                       help='目标宽度（像素）')
    parser.add_argument('--height', type=int,
                       help='目标高度（像素）')
    parser.add_argument('--interp', default='lanczos',
                       choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                       help='插值算法 (默认: lanczos)')
    parser.add_argument('-b', '--border', type=int, default=50,
                       help='边框大小 (默认: 50)')
    parser.add_argument('-f', '--format', default='png',
                       choices=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
                       help='输出格式 (默认: png)')
    parser.add_argument('-q', '--quality', type=int, default=95,
                       help='输出质量 1-100 (默认: 95)')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_DIR,
                       help='输出目录 (默认: output_upscaled)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归扫描子文件夹')
    parser.add_argument('--sharpen', action='store_true',
                       help='应用锐化')
    parser.add_argument('--denoise', action='store_true',
                       help='应用降噪')
    parser.add_argument('--ai-model', default='edsr',
                       choices=['edsr', 'espcn', 'fsrcnn', 'lapsrn'],
                       help='AI模型 (默认: edsr)')
    parser.add_argument('--ai-scale', type=int, default=2,
                       choices=[2, 3, 4, 8],
                       help='AI放大倍数 (默认: 2)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging()
    logging.info("图片放大处理工具启动")
    
    # 检查是否有命令行参数
    if len(sys.argv) == 1:
        # 交互模式
        interactive_mode()
        return
    
    # 命令行模式
    args = parse_arguments()
    
    # 创建配置
    config = ProcessingConfig(
        scale_factor=args.scale,
        target_width=args.width,
        target_height=args.height,
        interpolation=InterpolationMethod(args.interp),
        border_size=args.border,
        output_format=args.format,
        quality=args.quality,
        output_dir=args.output,
        ai_model=args.ai_model,
        ai_scale=args.ai_scale,
        sharpen=args.sharpen,
        denoise=args.denoise,
    )
    
    # 扫描图片
    images = scan_images(args.input, args.recursive)
    
    if not images:
        print("[ERR] 未找到支持的图片文件")
        logging.error("未找到图片文件")
        return
    
    print(f"找到 {len(images)} 张图片")
    
    # 确定处理方法
    method = UpscaleMethod(args.method)
    
    # 开始处理
    print("\n开始处理...")
    results = process_batch(images, method, config)
    
    # 生成报告
    report = generate_report(results, method, config)
    print("\n" + report)
    
    # 保存报告
    report_file = save_report(report)
    print(f"\n报告已保存: {report_file}")
    print(f"日志文件: {log_file}")
    
    # 保存配置
    save_config(config)
    
    logging.info("处理完成")


if __name__ == '__main__':
    main()
