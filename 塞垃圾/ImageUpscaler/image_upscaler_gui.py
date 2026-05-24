# -*- coding: utf-8 -*-
"""
图片放大处理工具 - 图形界面版本 v1.0
基于Tkinter的友好用户界面
"""

import os
import sys
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, Text, Scrollbar, 
    StringVar, IntVar, DoubleVar, BooleanVar,
    filedialog, messagebox, ttk, Menu
)

import numpy as np
from PIL import Image, ImageTk

# 导入核心处理模块
from image_upscaler import (
    ProcessingConfig, InterpolationMethod, UpscaleMethod,
    ProcessorFactory, scan_images, process_batch,
    generate_report, save_report, load_config, save_config,
    SUPPORTED_FORMATS, DEFAULT_OUTPUT_DIR
)


class ImageUpscalerGUI:
    """图片放大处理工具图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("图片放大处理工具 v1.0")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 配置
        self.config = load_config()
        self.input_paths = []
        self.processing = False
        
        # 创建界面
        self._create_menu()
        self._create_main_frame()
        self._create_input_section()
        self._create_method_section()
        self._create_params_section()
        self._create_preview_section()
        self._create_log_section()
        self._create_buttons()
        
        # 加载配置到界面
        self._load_config_to_ui()
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="添加文件", command=self._add_files)
        file_menu.add_command(label="添加文件夹", command=self._add_folder)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 帮助菜单
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self._show_help)
        help_menu.add_command(label="关于", command=self._show_about)
    
    def _create_main_frame(self):
        """创建主框架"""
        self.main_frame = Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill="both", expand=True)
        
        # 左侧配置面板
        self.left_frame = Frame(self.main_frame)
        self.left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 右侧预览和日志面板
        self.right_frame = Frame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True)
    
    def _create_input_section(self):
        """创建输入区域"""
        input_frame = Frame(self.left_frame)
        input_frame.pack(fill="x", pady=(0, 10))
        
        Label(input_frame, text="输入文件", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 文件列表
        list_frame = Frame(input_frame)
        list_frame.pack(fill="x", pady=5)
        
        self.file_listbox = Text(list_frame, height=6, width=35, wrap="none")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = Scrollbar(list_frame, command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        # 按钮
        btn_frame = Frame(input_frame)
        btn_frame.pack(fill="x")
        
        Button(btn_frame, text="添加文件", command=self._add_files).pack(side="left", padx=(0, 5))
        Button(btn_frame, text="添加文件夹", command=self._add_folder).pack(side="left", padx=(0, 5))
        Button(btn_frame, text="清空", command=self._clear_files).pack(side="left")
        
        # 文件计数
        self.file_count_var = StringVar(value="0 个文件")
        Label(input_frame, textvariable=self.file_count_var).pack(anchor="w")
    
    def _create_method_section(self):
        """创建处理方法选择区域"""
        method_frame = Frame(self.left_frame)
        method_frame.pack(fill="x", pady=(0, 10))
        
        Label(method_frame, text="放大方式", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.method_var = StringVar(value="interpolate")
        
        methods = [
            ("修改尺寸", "resize"),
            ("插值放大", "interpolate"),
            ("AI智能放大", "ai"),
            ("添加边框", "border"),
            ("组合所有", "all"),
        ]
        
        for text, value in methods:
            rb = ttk.Radiobutton(method_frame, text=text, variable=self.method_var, 
                                value=value, command=self._on_method_change)
            rb.pack(anchor="w", padx=20)
    
    def _create_params_section(self):
        """创建参数设置区域"""
        params_frame = Frame(self.left_frame)
        params_frame.pack(fill="x", pady=(0, 10))
        
        Label(params_frame, text="参数设置", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 放大倍数
        scale_frame = Frame(params_frame)
        scale_frame.pack(fill="x", pady=2)
        Label(scale_frame, text="放大倍数:").pack(side="left")
        self.scale_var = DoubleVar(value=2.0)
        Entry(scale_frame, textvariable=self.scale_var, width=8).pack(side="left", padx=5)
        
        # 插值算法
        interp_frame = Frame(params_frame)
        interp_frame.pack(fill="x", pady=2)
        Label(interp_frame, text="插值算法:").pack(side="left")
        self.interp_var = StringVar(value="lanczos")
        interp_combo = ttk.Combobox(interp_frame, textvariable=self.interp_var, 
                                   values=["nearest", "bilinear", "bicubic", "lanczos"],
                                   width=12, state="readonly")
        interp_combo.pack(side="left", padx=5)
        
        # 边框大小
        border_frame = Frame(params_frame)
        border_frame.pack(fill="x", pady=2)
        Label(border_frame, text="边框大小:").pack(side="left")
        self.border_var = IntVar(value=50)
        Entry(border_frame, textvariable=self.border_var, width=8).pack(side="left", padx=5)
        Label(border_frame, text="px").pack(side="left")
        
        # 输出格式
        format_frame = Frame(params_frame)
        format_frame.pack(fill="x", pady=2)
        Label(format_frame, text="输出格式:").pack(side="left")
        self.format_var = StringVar(value="png")
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var,
                                   values=["png", "jpg", "webp", "bmp"],
                                   width=8, state="readonly")
        format_combo.pack(side="left", padx=5)
        
        # 输出质量
        quality_frame = Frame(params_frame)
        quality_frame.pack(fill="x", pady=2)
        Label(quality_frame, text="输出质量:").pack(side="left")
        self.quality_var = IntVar(value=95)
        quality_scale = ttk.Scale(quality_frame, from_=1, to=100, variable=self.quality_var,
                                 orient="horizontal", length=100)
        quality_scale.pack(side="left", padx=5)
        self.quality_label = Label(quality_frame, text="95")
        self.quality_label.pack(side="left")
        self.quality_var.trace("w", lambda *args: self.quality_label.config(text=str(self.quality_var.get())))
        
        # 输出目录
        output_frame = Frame(params_frame)
        output_frame.pack(fill="x", pady=2)
        Label(output_frame, text="输出目录:").pack(side="left")
        self.output_var = StringVar(value=DEFAULT_OUTPUT_DIR)
        Entry(output_frame, textvariable=self.output_var, width=20).pack(side="left", padx=5)
        Button(output_frame, text="浏览", command=self._browse_output).pack(side="left")
        
        # 选项
        options_frame = Frame(params_frame)
        options_frame.pack(fill="x", pady=2)
        self.sharpen_var = BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="锐化", variable=self.sharpen_var).pack(side="left", padx=(0, 10))
        self.denoise_var = BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="降噪", variable=self.denoise_var).pack(side="left")
    
    def _create_preview_section(self):
        """创建预览区域"""
        preview_frame = Frame(self.right_frame)
        preview_frame.pack(fill="x", pady=(0, 10))
        
        Label(preview_frame, text="图片预览", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 预览画布
        self.preview_canvas = Label(preview_frame, bg="gray", width=50, height=15)
        self.preview_canvas.pack(fill="both", expand=True, pady=5)
        
        # 图片信息
        self.image_info_var = StringVar(value="未选择图片")
        Label(preview_frame, textvariable=self.image_info_var).pack(anchor="w")
    
    def _create_log_section(self):
        """创建日志区域"""
        log_frame = Frame(self.right_frame)
        log_frame.pack(fill="both", expand=True)
        
        Label(log_frame, text="处理日志", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 日志文本框
        log_text_frame = Frame(log_frame)
        log_text_frame.pack(fill="both", expand=True, pady=5)
        
        self.log_text = Text(log_text_frame, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        
        log_scrollbar = Scrollbar(log_text_frame, command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # 进度条
        self.progress_var = DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(log_frame, variable=self.progress_var, 
                                           maximum=100, length=400, mode="determinate")
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_var = StringVar(value="就绪")
        Label(log_frame, textvariable=self.status_var).pack(anchor="w")
    
    def _create_buttons(self):
        """创建按钮区域"""
        btn_frame = Frame(self.left_frame)
        btn_frame.pack(fill="x", pady=(10, 0))
        
        self.start_btn = Button(btn_frame, text="开始处理", command=self._start_processing,
                               bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.start_btn.pack(fill="x", pady=(0, 5))
        
        Button(btn_frame, text="保存配置", command=self._save_config).pack(fill="x", pady=(0, 5))
        Button(btn_frame, text="查看报告", command=self._view_report).pack(fill="x")
    
    def _load_config_to_ui(self):
        """加载配置到界面"""
        self.scale_var.set(self.config.scale_factor)
        self.interp_var.set(self.config.interpolation.value)
        self.border_var.set(self.config.border_size)
        self.format_var.set(self.config.output_format)
        self.quality_var.set(self.config.quality)
        self.output_var.set(self.config.output_dir)
        self.sharpen_var.set(self.config.sharpen)
        self.denoise_var.set(self.config.denoise)
    
    def _add_files(self):
        """添加文件"""
        files = filedialog.askopenfilenames(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.gif *.bmp *.webp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        if files:
            self.input_paths.extend(files)
            self._update_file_list()
    
    def _add_folder(self):
        """添加文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            images = scan_images(folder, recursive=True)
            self.input_paths.extend(images)
            self._update_file_list()
    
    def _clear_files(self):
        """清空文件列表"""
        self.input_paths = []
        self._update_file_list()
        self.preview_canvas.config(image="")
        self.image_info_var.set("未选择图片")
    
    def _update_file_list(self):
        """更新文件列表显示"""
        self.file_listbox.config(state="normal")
        self.file_listbox.delete("1.0", "end")
        
        for path in self.input_paths:
            self.file_listbox.insert("end", f"{Path(path).name}\n")
        
        self.file_listbox.config(state="disabled")
        self.file_count_var.set(f"{len(self.input_paths)} 个文件")
        
        # 预览第一张图片
        if self.input_paths:
            self._preview_image(self.input_paths[0])
    
    def _preview_image(self, path):
        """预览图片"""
        try:
            img = Image.open(path)
            
            # 缩放到预览区域
            max_size = (400, 200)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.preview_canvas.config(image=photo)
            self.preview_canvas.image = photo  # 保持引用
            
            self.image_info_var.set(f"{img.width}x{img.height} | {Path(path).name}")
        except Exception as e:
            self.image_info_var.set(f"预览失败: {e}")
    
    def _browse_output(self):
        """浏览输出目录"""
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            self.output_var.set(folder)
    
    def _on_method_change(self):
        """处理方法改变时"""
        method = self.method_var.get()
        # 可以根据方法启用/禁用某些控件
    
    def _log(self, message):
        """添加日志"""
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
    
    def _start_processing(self):
        """开始处理"""
        if not self.input_paths:
            messagebox.showwarning("警告", "请先添加图片文件")
            return
        
        if self.processing:
            messagebox.showwarning("警告", "正在处理中，请稍候")
            return
        
        # 获取配置
        config = ProcessingConfig(
            scale_factor=self.scale_var.get(),
            interpolation=InterpolationMethod(self.interp_var.get()),
            border_size=self.border_var.get(),
            output_format=self.format_var.get(),
            quality=self.quality_var.get(),
            output_dir=self.output_var.get(),
            sharpen=self.sharpen_var.get(),
            denoise=self.denoise_var.get(),
        )
        
        method = UpscaleMethod(self.method_var.get())
        
        # 开始处理线程
        self.processing = True
        self.start_btn.config(state="disabled", text="处理中...")
        self.status_var.set("正在处理...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self._process_thread, args=(method, config))
        thread.daemon = True
        thread.start()
    
    def _process_thread(self, method, config):
        """处理线程"""
        try:
            def progress_callback(current, total, result):
                progress = (current / total) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"处理中... {current}/{total}")
                
                if result['status'] == 'success':
                    self._log(f"[OK] {result['message']}")
                else:
                    self._log(f"[ERR] {result['message']}")
            
            results = process_batch(self.input_paths, method, config, progress_callback)
            
            # 生成报告
            report = generate_report(results, method, config)
            self._log("\n" + "=" * 50)
            self._log(report)
            
            # 保存报告
            report_file = save_report(report)
            self._log(f"\n报告已保存: {report_file}")
            
            # 保存配置
            save_config(config)
            
            self.status_var.set("处理完成")
            messagebox.showinfo("完成", f"处理完成！\n成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
            
        except Exception as e:
            self._log(f"错误: {e}")
            self.status_var.set("处理失败")
            messagebox.showerror("错误", f"处理失败: {e}")
        
        finally:
            self.processing = False
            self.start_btn.config(state="normal", text="开始处理")
            self.progress_var.set(0)
    
    def _save_config(self):
        """保存配置"""
        config = ProcessingConfig(
            scale_factor=self.scale_var.get(),
            interpolation=InterpolationMethod(self.interp_var.get()),
            border_size=self.border_var.get(),
            output_format=self.format_var.get(),
            quality=self.quality_var.get(),
            output_dir=self.output_var.get(),
            sharpen=self.sharpen_var.get(),
            denoise=self.denoise_var.get(),
        )
        save_config(config)
        messagebox.showinfo("成功", "配置已保存")
    
    def _view_report(self):
        """查看报告"""
        report_dir = Path("reports")
        if report_dir.exists():
            reports = sorted(report_dir.glob("report_*.txt"), reverse=True)
            if reports:
                import subprocess
                subprocess.Popen(["notepad", str(reports[0])])
            else:
                messagebox.showinfo("提示", "暂无报告文件")
        else:
            messagebox.showinfo("提示", "暂无报告文件")
    
    def _show_help(self):
        """显示帮助"""
        help_text = """
图片放大处理工具使用说明

支持的处理方式：
1. 修改尺寸 - 直接调整图片像素大小
2. 插值放大 - 使用插值算法放大（最近邻/双线性/双三次/Lanczos）
3. AI智能放大 - 使用深度学习超分辨率算法
4. 添加边框 - 保持原图内容，添加边框使整体变大
5. 组合所有 - 依次应用所有处理方式

支持的格式：JPG、PNG、GIF、BMP、WEBP、TIFF

提示：
- 可以添加单个文件或整个文件夹
- 支持批量处理
- 处理过程中可以查看进度和日志
        """
        messagebox.showinfo("使用说明", help_text)
    
    def _show_about(self):
        """显示关于"""
        messagebox.showinfo("关于", "图片放大处理工具 v1.0\n\n支持多种图片格式的放大处理")


def main():
    """主函数"""
    root = Tk()
    app = ImageUpscalerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
