"""
代码生成器

负责将算子配置转换为可执行的代码。
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import asdict

from .operator_base import OperatorBase, Backend, DataType
from .backend_manager import BackendManager


class CodeTemplate:
    """代码模板管理"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 注册自定义过滤器
        self.env.filters['dtype_to_c'] = self._dtype_to_c_type
        self.env.filters['shape_size'] = self._calculate_shape_size
        
    def _dtype_to_c_type(self, dtype: DataType) -> str:
        """将数据类型转换为C类型"""
        mapping = {
            DataType.FLOAT32: "float",
            DataType.FLOAT16: "half",
            DataType.INT32: "int",
            DataType.INT64: "long long",
            DataType.BOOL: "bool"
        }
        return mapping.get(dtype, "float")
    
    def _calculate_shape_size(self, shape_dims: List[Union[int, str]]) -> str:
        """计算形状大小的表达式"""
        size_expr = []
        for dim in shape_dims:
            if isinstance(dim, int):
                size_expr.append(str(dim))
            else:
                size_expr.append(str(dim))
        return " * ".join(size_expr)
    
    def get_template(self, template_name: str) -> Template:
        """获取模板"""
        return self.env.get_template(template_name)
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """渲染模板"""
        template = self.get_template(template_name)
        return template.render(**context)


class CodeGenerator:
    """代码生成器主类"""
    
    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            # 使用默认模板目录
            current_dir = Path(__file__).parent.parent
            template_dir = current_dir / "templates"
            
        self.template_manager = CodeTemplate(str(template_dir))
        self.backend_manager = BackendManager()
        
    def generate_operator_code(self, operator: OperatorBase) -> Dict[str, str]:
        """生成算子完整代码"""
        backend = operator.config.backend
        operator_type = operator.get_operator_type().lower()
        
        # 准备模板上下文
        context = self._prepare_context(operator)
        
        # 生成不同文件的代码
        generated_files = {}
        
        # 1. 生成头文件
        header_template = f"{backend.value}/{operator_type}_kernel.h"
        if self._template_exists(header_template):
            generated_files["header"] = self.template_manager.render_template(
                header_template, context
            )
        
        # 2. 生成实现文件
        impl_template = f"{backend.value}/{operator_type}_kernel.{self._get_impl_extension(backend)}"
        if self._template_exists(impl_template):
            generated_files["implementation"] = self.template_manager.render_template(
                impl_template, context
            )
        
        # 3. 生成Python绑定
        python_template = f"python/{operator_type}_binding.py"
        if self._template_exists(python_template):
            generated_files["python_binding"] = self.template_manager.render_template(
                python_template, context
            )
        
        # 4. 生成测试文件
        test_template = f"tests/{operator_type}_test.py"
        if self._template_exists(test_template):
            generated_files["test"] = self.template_manager.render_template(
                test_template, context
            )
        
        # 5. 生成构建脚本
        build_template = f"build/{backend.value}_build.cmake"
        if self._template_exists(build_template):
            generated_files["build_script"] = self.template_manager.render_template(
                build_template, context
            )
        
        return generated_files
    
    def _prepare_context(self, operator: OperatorBase) -> Dict[str, Any]:
        """准备模板渲染上下文"""
        backend_cap = self.backend_manager.get_backend_capability(operator.config.backend)
        
        context = {
            # 算子基本信息
            "operator": operator,
            "operator_type": operator.get_operator_type(),
            "kernel_name": operator.get_kernel_name(),
            
            # 配置信息
            "config": operator.config,
            "backend": operator.config.backend,
            "dtype": operator.config.dtype,
            
            # 形状和参数
            "input_shapes": operator.input_shapes,
            "output_shapes": operator.output_shapes,
            "parameters": operator.parameters,
            
            # 后端能力
            "backend_capability": backend_cap,
            
            # 依赖和编译标志
            "dependencies": operator.get_dependencies(),
            "compile_flags": self.backend_manager.get_backend_specific_flags(operator.config.backend),
            
            # 性能优化
            "optimization_level": operator.config.optimization_level,
            "debug_mode": operator.config.debug_mode,
            
            # 工具函数
            "dtype_to_c": self.template_manager._dtype_to_c_type,
            "shape_size": self.template_manager._calculate_shape_size,
        }
        
        # 添加后端特定的上下文
        if operator.config.backend == Backend.CUDA:
            context.update(self._get_cuda_context(operator))
        elif operator.config.backend == Backend.CPU:
            context.update(self._get_cpu_context(operator))
        
        return context
    
    def _get_cuda_context(self, operator: OperatorBase) -> Dict[str, Any]:
        """获取CUDA特定上下文"""
        # 计算推荐的线程块配置
        total_elements = 1
        for shape in operator.output_shapes:
            for dim in shape.dims:
                if isinstance(dim, int):
                    total_elements *= dim
        
        block_size = self.backend_manager.get_recommended_block_size(
            Backend.CUDA, total_elements
        )
        
        grid_size = (total_elements + block_size - 1) // block_size
        
        # 安全获取CUDA架构信息
        cuda_cap = self.backend_manager.get_backend_capability(Backend.CUDA)
        cuda_arch = cuda_cap.compute_capability if cuda_cap and cuda_cap.compute_capability else "7.5"
        
        return {
            "block_size": block_size,
            "grid_size": grid_size,
            "shared_memory_size": 0,  # 可以根据需要计算
            "cuda_arch": cuda_arch,
        }
    
    def _get_cpu_context(self, operator: OperatorBase) -> Dict[str, Any]:
        """获取CPU特定上下文"""
        num_threads = self.backend_manager.get_recommended_block_size(Backend.CPU, 0)
        
        return {
            "num_threads": num_threads,
            "use_openmp": True,
            "vectorization": operator.config.optimization_level >= 1,
        }
    
    def _template_exists(self, template_name: str) -> bool:
        """检查模板是否存在"""
        template_path = self.template_manager.template_dir / template_name
        return template_path.exists()
    
    def _get_impl_extension(self, backend: Backend) -> str:
        """获取实现文件扩展名"""
        if backend == Backend.CUDA:
            return "cu"
        elif backend == Backend.CPU:
            return "cpp"
        elif backend == Backend.TRITON:
            return "py"
        elif backend == Backend.ROCM:
            return "cpp"
        elif backend == Backend.OPENCL:
            return "cl"
        else:
            return "cpp"
    
    def save_generated_code(self, generated_files: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """保存生成的代码到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for file_type, content in generated_files.items():
            # 根据文件类型确定文件名
            filename = self._get_output_filename(file_type)
            file_path = output_path / filename
            
            # 创建子目录
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            saved_files[file_type] = str(file_path)
        
        return saved_files
    
    def _get_output_filename(self, file_type: str) -> str:
        """根据文件类型获取输出文件名"""
        filename_mapping = {
            "header": "kernel.h",
            "implementation": "kernel.py",  # 默认，会根据后端调整
            "python_binding": "binding.py",
            "test": "test.py",
            "build_script": "CMakeLists.txt",
        }
        return filename_mapping.get(file_type, f"{file_type}.txt")
    
    def generate_makefile(self, operator: OperatorBase, output_dir: str) -> str:
        """生成Makefile"""
        context = self._prepare_context(operator)
        
        makefile_template = f"build/{operator.config.backend.value}_makefile"
        if self._template_exists(makefile_template):
            makefile_content = self.template_manager.render_template(
                makefile_template, context
            )
            
            makefile_path = Path(output_dir) / "Makefile"
            with open(makefile_path, 'w') as f:
                f.write(makefile_content)
            
            return str(makefile_path)
        
        return ""