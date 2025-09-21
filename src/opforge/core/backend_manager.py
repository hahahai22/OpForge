"""
后端管理器

管理不同计算后端的配置和功能支持。
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import subprocess
import sys

from .operator_base import Backend


@dataclass
class BackendCapability:
    """后端能力描述"""
    name: str
    version: str
    supports_fp16: bool = False
    supports_int8: bool = False
    max_shared_memory: int = 0  # 最大共享内存 (bytes)
    max_threads_per_block: int = 0
    compute_capability: str = ""  # CUDA计算能力
    
    
class BackendManager:
    """后端管理器"""
    
    def __init__(self):
        self._backend_capabilities: Dict[Backend, BackendCapability] = {}
        self._available_backends: List[Backend] = []
        self._detect_available_backends()
    
    def _detect_available_backends(self) -> None:
        """检测可用的后端"""
        # 检测CPU后端（总是可用）
        self._available_backends.append(Backend.CPU)
        self._backend_capabilities[Backend.CPU] = BackendCapability(
            name="CPU",
            version="1.0.0",
            supports_fp16=True,
            supports_int8=True,
        )
        
        # 检测CUDA后端
        if self._check_cuda_available():
            self._available_backends.append(Backend.CUDA)
            cuda_info = self._get_cuda_info()
            self._backend_capabilities[Backend.CUDA] = cuda_info
            
        # 检测Triton后端
        if self._check_triton_available():
            self._available_backends.append(Backend.TRITON)
            triton_info = self._get_triton_info()
            self._backend_capabilities[Backend.TRITON] = triton_info
            
        # 检测ROCm后端
        if self._check_rocm_available():
            self._available_backends.append(Backend.ROCM)
            rocm_info = self._get_rocm_info()
            self._backend_capabilities[Backend.ROCM] = rocm_info
    
    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                # 尝试使用nvidia-smi命令
                result = subprocess.run(['nvidia-smi'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def _get_cuda_info(self) -> BackendCapability:
        """获取CUDA设备信息"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.get_device_properties(0)
                return BackendCapability(
                    name="CUDA",
                    version=torch.version.cuda or "unknown",
                    supports_fp16=True,
                    supports_int8=True,
                    max_shared_memory=device.total_memory,
                    max_threads_per_block=device.max_threads_per_block,
                    compute_capability=f"{device.major}.{device.minor}"
                )
        except ImportError:
            pass
            
        return BackendCapability(
            name="CUDA",
            version="unknown",
            supports_fp16=True,
            supports_int8=True,
        )
    
    def _check_triton_available(self) -> bool:
        """检查Triton是否可用"""
        try:
            import triton
            return True
        except ImportError:
            return False
    
    def _get_triton_info(self) -> BackendCapability:
        """获取Triton信息"""
        try:
            import triton
            return BackendCapability(
                name="Triton",
                version=getattr(triton, '__version__', "unknown"),
                supports_fp16=True,
                supports_int8=True,
                compute_capability="triton_jit"
            )
        except ImportError:
            return BackendCapability(
                name="Triton",
                version="unknown",
                supports_fp16=True,
                supports_int8=True,
            )
    
    def _check_rocm_available(self) -> bool:
        """检查ROCm是否可用"""
        try:
            import torch
            return torch.version.hip is not None
        except (ImportError, AttributeError):
            try:
                # 尝试使用rocm-smi命令
                result = subprocess.run(['rocm-smi'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def _get_rocm_info(self) -> BackendCapability:
        """获取ROCm设备信息"""
        try:
            import torch
            if torch.version.hip:
                return BackendCapability(
                    name="ROCm",
                    version=torch.version.hip,
                    supports_fp16=True,
                    supports_int8=True,
                    compute_capability="gfx90a"  # 默认架构
                )
        except ImportError:
            pass
            
        return BackendCapability(
            name="ROCm",
            version="unknown",
            supports_fp16=True,
            supports_int8=True,
        )
    
    def get_available_backends(self) -> List[Backend]:
        """获取可用的后端列表"""
        return self._available_backends.copy()
    
    def is_backend_available(self, backend: Backend) -> bool:
        """检查指定后端是否可用"""
        return backend in self._available_backends
    
    def get_backend_capability(self, backend: Backend) -> Optional[BackendCapability]:
        """获取后端能力信息"""
        return self._backend_capabilities.get(backend)
    
    def get_optimal_backend(self, requirements: Dict[str, Any] = None) -> Backend:
        """根据需求获取最优后端"""
        if requirements is None:
            requirements = {}
        
        # 如果有CUDA，默认使用CUDA
        if Backend.CUDA in self._available_backends:
            cuda_cap = self._backend_capabilities[Backend.CUDA]
            
            # 检查是否满足特殊需求
            if requirements.get("requires_fp16", False) and not cuda_cap.supports_fp16:
                pass  # 继续寻找其他后端
            else:
                return Backend.CUDA
        
        # 回退到CPU
        if Backend.CPU in self._available_backends:
            return Backend.CPU
            
        # 如果没有可用后端，抛出异常
        raise RuntimeError("没有可用的计算后端")
    
    def validate_backend_for_operator(self, backend: Backend, operator_type: str) -> bool:
        """验证后端是否支持指定算子"""
        if not self.is_backend_available(backend):
            return False
            
        # 这里可以根据算子类型和后端能力进行更详细的验证
        # 例如：某些算子可能只在特定的CUDA版本上支持
        
        return True
    
    def get_backend_specific_flags(self, backend: Backend) -> List[str]:
        """获取后端特定的编译标志"""
        flags = []
        
        if backend == Backend.CUDA:
            cuda_cap = self._backend_capabilities.get(Backend.CUDA)
            if cuda_cap and cuda_cap.compute_capability:
                flags.extend([
                    f"-arch=sm_{cuda_cap.compute_capability.replace('.', '')}",
                    "-O3",
                    "--use_fast_math"
                ])
        elif backend == Backend.CPU:
            flags.extend([
                "-O3",
                "-march=native",
                "-fopenmp"
            ])
        elif backend == Backend.OPENCL:
            flags.extend([
                "-cl-fast-relaxed-math",
                "-cl-mad-enable"
            ])
            
        return flags
    
    def get_recommended_block_size(self, backend: Backend, problem_size: int) -> int:
        """获取推荐的块大小"""
        if backend == Backend.CUDA:
            cuda_cap = self._backend_capabilities.get(Backend.CUDA)
            if cuda_cap and cuda_cap.max_threads_per_block:
                # 根据问题大小和硬件能力选择合适的块大小
                if problem_size < 256:
                    return min(problem_size, 64)
                elif problem_size < 1024:
                    return min(problem_size, 256)
                else:
                    return min(cuda_cap.max_threads_per_block, 1024)
            return 256  # 默认块大小
        
        elif backend == Backend.CPU:
            # CPU使用线程数而不是块大小
            try:
                import multiprocessing
                return multiprocessing.cpu_count()
            except:
                return 4  # 默认4线程
                
        return 1  # 其他后端默认