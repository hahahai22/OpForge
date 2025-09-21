"""
算子基类定义

所有算子生成器的基础抽象类，定义了通用的接口和行为。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum


class DataType(Enum):
    """支持的数据类型"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


class Backend(Enum):
    """支持的后端类型"""
    CUDA = "cuda"
    CPU = "cpu"
    OPENCL = "opencl"
    VULKAN = "vulkan"
    TRITON = "triton"
    ROCM = "rocm"


class HardwareInstruction(Enum):
    """支持的硬件指令类型"""
    # NVIDIA GPU指令
    TENSOR_CORE = "tensorcore"        # Tensor Core指令
    WMMA = "wmma"                    # Warp Matrix Multiply Accumulate
    CUDA_CORE = "cuda_core"          # 标准CUDA Core
    
    # 点积指令
    DOT2 = "dot2"                    # 2元素点积
    DOT4 = "dot4"                    # 4元素点积
    DP4A = "dp4a"                    # 4x8bit点积累加
    
    # 内存访问指令
    BUFFER_LOAD = "buffer_load"      # 缓冲区加载
    BUFFER_STORE = "buffer_store"    # 缓冲区存储
    SHARED_LOAD = "shared_load"      # 共享内存加载
    GLOBAL_LOAD = "global_load"      # 全局内存加载
    
    # AMD GPU指令
    MFMA = "mfma"                    # Matrix Fused Multiply Add
    DOT_MIXED = "dot_mixed"          # 混合精度点积
    
    # 特殊指令
    SHUFFLE = "shuffle"              # Warp shuffle指令
    BALLOT = "ballot"                # 投票指令
    REDUCE = "reduce"                # 归约指令


@dataclass
class HardwareConfig:
    """硬件指令配置"""
    # 支持的指令集
    supported_instructions: Set[HardwareInstruction] = field(default_factory=set)
    
    # Tensor Core配置
    tensor_core_shapes: List[tuple] = field(default_factory=lambda: [(16, 16, 16), (8, 32, 16)])
    tensor_core_dtypes: Set[DataType] = field(default_factory=lambda: {DataType.FLOAT16, DataType.FLOAT32})
    
    # 点积指令配置
    dot_precision: Dict[str, int] = field(default_factory=lambda: {"dot2": 2, "dot4": 4})
    
    # 内存层次配置
    shared_memory_size: int = 48 * 1024  # 48KB
    global_memory_bandwidth: float = 900.0  # GB/s
    
    # 特殊功能
    warp_size: int = 32
    max_threads_per_block: int = 1024
    
    # 自定义指令参数
    custom_instruction_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorShape:
    """张量形状描述"""
    dims: List[Union[int, str]]  # 支持动态维度，如 ["N", "C", 512, 512]
    dtype: DataType
    name: str = ""
    
    def __str__(self) -> str:
        dims_str = "x".join(str(d) for d in self.dims)
        return f"{self.name}[{dims_str}:{self.dtype.value}]"


@dataclass 
class OperatorConfig:
    """算子配置基类"""
    name: str
    backend: Backend
    dtype: DataType = DataType.FLOAT32
    optimization_level: int = 2  # 0: 无优化, 1: 基础优化, 2: 激进优化
    debug_mode: bool = False
    custom_attrs: Dict[str, Any] = None
    
    # 硬件指令配置
    hardware_config: Optional[HardwareConfig] = None
    enable_tensor_core: bool = False
    enable_dot_instructions: bool = False
    enable_buffer_ops: bool = False
    
    def __post_init__(self):
        if self.custom_attrs is None:
            self.custom_attrs = {}
        
        # 默认硬件配置
        if self.hardware_config is None:
            self.hardware_config = HardwareConfig()
            
        # 根据后端自动配置硬件指令
        self._configure_hardware_instructions()
    
    def _configure_hardware_instructions(self):
        """根据后端类型自动配置硬件指令"""
        if self.backend == Backend.CUDA:
            self.hardware_config.supported_instructions.update({
                HardwareInstruction.TENSOR_CORE,
                HardwareInstruction.WMMA,
                HardwareInstruction.CUDA_CORE,
                HardwareInstruction.DOT2,
                HardwareInstruction.DOT4,
                HardwareInstruction.DP4A,
                HardwareInstruction.BUFFER_LOAD,
                HardwareInstruction.BUFFER_STORE,
                HardwareInstruction.SHARED_LOAD,
                HardwareInstruction.GLOBAL_LOAD,
                HardwareInstruction.SHUFFLE,
                HardwareInstruction.BALLOT,
                HardwareInstruction.REDUCE
            })
        elif self.backend == Backend.ROCM:
            self.hardware_config.supported_instructions.update({
                HardwareInstruction.MFMA,
                HardwareInstruction.DOT_MIXED,
                HardwareInstruction.BUFFER_LOAD,
                HardwareInstruction.BUFFER_STORE,
                HardwareInstruction.SHARED_LOAD,
                HardwareInstruction.GLOBAL_LOAD
            })
        elif self.backend == Backend.TRITON:
            # Triton支持高级抽象的硬件指令
            self.hardware_config.supported_instructions.update({
                HardwareInstruction.TENSOR_CORE,
                HardwareInstruction.DOT2,
                HardwareInstruction.DOT4,
                HardwareInstruction.BUFFER_LOAD,
                HardwareInstruction.BUFFER_STORE
            })
    
    def enable_instruction(self, instruction: HardwareInstruction) -> bool:
        """启用特定硬件指令"""
        if instruction in self.hardware_config.supported_instructions:
            if instruction == HardwareInstruction.TENSOR_CORE:
                self.enable_tensor_core = True
            elif instruction in [HardwareInstruction.DOT2, HardwareInstruction.DOT4, HardwareInstruction.DP4A]:
                self.enable_dot_instructions = True
            elif instruction in [HardwareInstruction.BUFFER_LOAD, HardwareInstruction.BUFFER_STORE]:
                self.enable_buffer_ops = True
            return True
        return False
    
    def get_enabled_instructions(self) -> Set[HardwareInstruction]:
        """获取已启用的硬件指令"""
        enabled = set()
        if self.enable_tensor_core:
            enabled.add(HardwareInstruction.TENSOR_CORE)
        if self.enable_dot_instructions:
            enabled.update({HardwareInstruction.DOT2, HardwareInstruction.DOT4})
        if self.enable_buffer_ops:
            enabled.update({HardwareInstruction.BUFFER_LOAD, HardwareInstruction.BUFFER_STORE})
        return enabled


class OperatorBase(ABC):
    """算子生成器基类"""
    
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.input_shapes: List[TensorShape] = []
        self.output_shapes: List[TensorShape] = []
        self.parameters: Dict[str, Any] = {}
        
    @abstractmethod
    def get_operator_type(self) -> str:
        """返回算子类型名称"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        pass
    
    @abstractmethod
    def infer_output_shape(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """根据输入形状推断输出形状"""
        pass
    
    @abstractmethod
    def generate_forward_code(self) -> str:
        """生成前向传播代码"""
        pass
    
    @abstractmethod
    def generate_backward_code(self) -> str:
        """生成反向传播代码"""
        pass
        
    def generate_header_code(self) -> str:
        """生成头文件代码"""
        return ""
    
    def generate_test_code(self) -> str:
        """生成测试代码"""
        return ""
    
    def get_dependencies(self) -> List[str]:
        """返回依赖的库或头文件"""
        base_deps = []
        if self.config.backend == Backend.CUDA:
            base_deps.extend(["cuda_runtime.h", "cudnn.h"])
        elif self.config.backend == Backend.CPU:
            base_deps.extend(["cblas.h", "omp.h"])
        return base_deps
    
    def get_kernel_name(self) -> str:
        """获取内核函数名称"""
        return f"{self.get_operator_type().lower()}_{self.config.backend.value}_kernel"
    
    def set_input_shapes(self, shapes: List[TensorShape]) -> None:
        """设置输入张量形状"""
        self.input_shapes = shapes
        self.output_shapes = self.infer_output_shape(shapes)
    
    def add_parameter(self, name: str, value: Any) -> None:
        """添加算子参数"""
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """获取算子参数"""
        return self.parameters.get(name, default)
    
    def get_memory_requirements(self) -> Dict[str, int]:
        """计算内存需求"""
        input_memory = sum(self._calculate_tensor_size(shape) for shape in self.input_shapes)
        output_memory = sum(self._calculate_tensor_size(shape) for shape in self.output_shapes)
        
        return {
            "input_memory_bytes": input_memory,
            "output_memory_bytes": output_memory,
            "total_memory_bytes": input_memory + output_memory
        }
    
    def _calculate_tensor_size(self, shape: TensorShape) -> int:
        """计算张量大小（字节）"""
        dtype_sizes = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.BOOL: 1
        }
        
        elements = 1
        for dim in shape.dims:
            if isinstance(dim, int):
                elements *= dim
            else:
                # 对于动态维度，使用默认估计值
                elements *= 1024
                
        return elements * dtype_sizes[shape.dtype]
    
    def get_performance_hints(self) -> List[str]:
        """获取性能优化建议"""
        hints = []
        
        # 通用性能建议
        if self.config.optimization_level == 0:
            hints.append("建议启用优化以获得更好的性能")
            
        # 内存使用建议
        memory_req = self.get_memory_requirements()
        if memory_req["total_memory_bytes"] > 1024 * 1024 * 1024:  # > 1GB
            hints.append("内存使用量较大，考虑使用内存优化策略")
            
        return hints