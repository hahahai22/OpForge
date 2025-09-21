"""
2D卷积算子生成器

支持生成高性能的2D卷积算子，包括标准卷积、深度可分离卷积、分组卷积等变体。
"""

from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import math

from ..core.operator_base import OperatorBase, OperatorConfig, TensorShape, DataType, Backend


@dataclass
class Conv2DConfig(OperatorConfig):
    """2D卷积配置"""
    # 基本参数（使用None作为默认值，在__post_init__中验证）
    in_channels: int = None
    out_channels: int = None
    kernel_size: Union[int, tuple] = 3
    stride: Union[int, tuple] = 1
    padding: Union[int, tuple, str] = 0
    dilation: Union[int, tuple] = 1
    groups: int = 1
    
    # 可选参数
    bias: bool = True
    padding_mode: str = "zeros"  # "zeros", "reflect", "replicate", "circular"
    
    # 性能优化选项
    use_winograd: bool = False
    use_fft: bool = False
    use_im2col: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        
        # 验证必需参数
        if self.in_channels is None:
            raise ValueError("in_channels必须指定")
        if self.out_channels is None:
            raise ValueError("out_channels必须指定")
        
        # 标准化参数格式
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        elif isinstance(self.padding, str):
            if self.padding == "same":
                # 计算same padding
                kh, kw = self.kernel_size
                self.padding = ((kh - 1) // 2, (kw - 1) // 2)
            elif self.padding == "valid":
                self.padding = (0, 0)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)
        
        # 验证分组卷积参数
        if self.in_channels % self.groups != 0:
            raise ValueError(f"输入通道数 {self.in_channels} 必须能被组数 {self.groups} 整除")
        if self.out_channels % self.groups != 0:
            raise ValueError(f"输出通道数 {self.out_channels} 必须能被组数 {self.groups} 整除")


class Conv2DOperator(OperatorBase):
    """2D卷积算子生成器"""
    
    def __init__(self, config: Conv2DConfig):
        super().__init__(config)
        self.conv_config = config
        
        # 设置基本参数
        self.add_parameter("in_channels", config.in_channels)
        self.add_parameter("out_channels", config.out_channels)
        self.add_parameter("kernel_h", config.kernel_size[0])
        self.add_parameter("kernel_w", config.kernel_size[1])
        self.add_parameter("stride_h", config.stride[0])
        self.add_parameter("stride_w", config.stride[1])
        self.add_parameter("pad_h", config.padding[0])
        self.add_parameter("pad_w", config.padding[1])
        self.add_parameter("dilation_h", config.dilation[0])
        self.add_parameter("dilation_w", config.dilation[1])
        self.add_parameter("groups", config.groups)
        self.add_parameter("use_bias", config.bias)
    
    def get_operator_type(self) -> str:
        """返回算子类型"""
        if self.conv_config.groups == self.conv_config.in_channels:
            return "depthwise_conv2d"
        elif self.conv_config.groups > 1:
            return "group_conv2d"
        else:
            return "conv2d"
    
    def validate_config(self) -> bool:
        """验证配置"""
        config = self.conv_config
        
        # 检查基本参数
        if config.in_channels <= 0 or config.out_channels <= 0:
            return False
        
        if any(k <= 0 for k in config.kernel_size):
            return False
            
        if any(s <= 0 for s in config.stride):
            return False
            
        if any(d <= 0 for d in config.dilation):
            return False
            
        if config.groups <= 0:
            return False
        
        # 检查算法兼容性
        if config.use_winograd:
            # Winograd只支持特定的卷积核大小
            if config.kernel_size not in [(3, 3), (5, 5)]:
                return False
            if config.dilation != (1, 1):
                return False
        
        return True
    
    def infer_output_shape(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """推断输出形状"""
        if not input_shapes:
            raise ValueError("需要至少一个输入张量")
        
        input_shape = input_shapes[0]
        if len(input_shape.dims) != 4:
            raise ValueError("输入张量必须是4D (N, C, H, W)")
        
        N, C_in, H_in, W_in = input_shape.dims
        
        # 验证输入通道数
        if isinstance(C_in, int) and C_in != self.conv_config.in_channels:
            raise ValueError(f"输入通道数不匹配: 期望 {self.conv_config.in_channels}, 实际 {C_in}")
        
        # 计算输出尺寸
        kh, kw = self.conv_config.kernel_size
        sh, sw = self.conv_config.stride
        ph, pw = self.conv_config.padding
        dh, dw = self.conv_config.dilation
        
        def calc_output_size(input_size, kernel_size, stride, padding, dilation):
            if isinstance(input_size, str):
                # 动态维度，返回表达式
                return f"({input_size} + 2*{padding} - {dilation}*({kernel_size} - 1) - 1) // {stride} + 1"
            else:
                return (input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1
        
        H_out = calc_output_size(H_in, kh, sh, ph, dh)
        W_out = calc_output_size(W_in, kw, sw, pw, dw)
        
        output_shape = TensorShape(
            dims=[N, self.conv_config.out_channels, H_out, W_out],
            dtype=input_shape.dtype,
            name="output"
        )
        
        return [output_shape]
    
    def generate_forward_code(self) -> str:
        """生成前向传播代码"""
        if self.config.backend == Backend.CUDA:
            return self._generate_cuda_forward()
        elif self.config.backend == Backend.CPU:
            return self._generate_cpu_forward()
        else:
            raise NotImplementedError(f"后端 {self.config.backend} 尚未支持")
    
    def generate_backward_code(self) -> str:
        """生成反向传播代码"""
        if self.config.backend == Backend.CUDA:
            return self._generate_cuda_backward()
        elif self.config.backend == Backend.CPU:
            return self._generate_cpu_backward()
        else:
            raise NotImplementedError(f"后端 {self.config.backend} 尚未支持")
    
    def _generate_cuda_forward(self) -> str:
        """生成CUDA前向代码"""
        # 选择最优算法
        algorithm = self._select_cuda_algorithm()
        
        code_sections = []
        
        if algorithm == "direct":
            code_sections.append(self._generate_cuda_direct_conv())
        elif algorithm == "im2col":
            code_sections.append(self._generate_cuda_im2col_conv())
        elif algorithm == "winograd":
            code_sections.append(self._generate_cuda_winograd_conv())
        elif algorithm == "fft":
            code_sections.append(self._generate_cuda_fft_conv())
        
        return "\n\n".join(code_sections)
    
    def _select_cuda_algorithm(self) -> str:
        """选择CUDA算法"""
        config = self.conv_config
        
        # 如果用户指定了算法
        if config.use_winograd and self._can_use_winograd():
            return "winograd"
        elif config.use_fft and self._can_use_fft():
            return "fft"
        elif config.use_im2col:
            return "im2col"
        
        # 自动选择算法
        kh, kw = config.kernel_size
        
        # 小卷积核使用直接卷积
        if kh <= 3 and kw <= 3:
            return "direct"
        
        # 大卷积核使用im2col + GEMM
        if kh >= 5 or kw >= 5:
            return "im2col"
        
        # 默认使用直接卷积
        return "direct"
    
    def _can_use_winograd(self) -> bool:
        """检查是否可以使用Winograd"""
        config = self.conv_config
        return (config.kernel_size in [(3, 3), (5, 5)] and 
                config.dilation == (1, 1) and
                config.groups == 1)
    
    def _can_use_fft(self) -> bool:
        """检查是否可以使用FFT"""
        config = self.conv_config
        kh, kw = config.kernel_size
        return kh >= 7 or kw >= 7  # 大卷积核适合FFT
    
    def _generate_cuda_direct_conv(self) -> str:
        """生成CUDA直接卷积代码"""
        return f"""
// CUDA直接卷积实现
__global__ void {self.get_kernel_name()}_direct(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * H_out * W_out;
    
    if (idx >= total_threads) return;
    
    // 解析输出位置
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);
    
    float sum = 0.0f;
    
    int group_size_in = C_in / {self.conv_config.groups};
    int group_size_out = C_out / {self.conv_config.groups};
    int group_id = c_out / group_size_out;
    int c_in_start = group_id * group_size_in;
    int c_in_end = c_in_start + group_size_in;
    
    // 卷积计算
    for (int c_in = c_in_start; c_in < c_in_end; c_in++) {{
        for (int k_h = 0; k_h < {self.conv_config.kernel_size[0]}; k_h++) {{
            for (int k_w = 0; k_w < {self.conv_config.kernel_size[1]}; k_w++) {{
                int h_in = h_out * {self.conv_config.stride[0]} - {self.conv_config.padding[0]} + 
                          k_h * {self.conv_config.dilation[0]};
                int w_in = w_out * {self.conv_config.stride[1]} - {self.conv_config.padding[1]} + 
                          k_w * {self.conv_config.dilation[1]};
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {{
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out * group_size_in + (c_in - c_in_start)) * 
                                    {self.conv_config.kernel_size[0]} + k_h) * 
                                    {self.conv_config.kernel_size[1]} + k_w;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }}
            }}
        }}
    }}
    
    {"// 添加偏置" if self.conv_config.bias else "// 无偏置"}
    {"if (bias != nullptr) sum += bias[c_out];" if self.conv_config.bias else ""}
    
    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}}
"""
    
    def _generate_cuda_im2col_conv(self) -> str:
        """生成CUDA im2col+GEMM卷积代码"""
        return f"""
// CUDA im2col + GEMM卷积实现
__global__ void {self.get_kernel_name()}_im2col(
    const float* input,
    float* data_col,
    int N, int C_in, int H_in, int W_in,
    int H_out, int W_out
) {{
    // im2col核函数实现
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_in * {self.conv_config.kernel_size[0]} * {self.conv_config.kernel_size[1]} * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    // 解析位置索引
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k_w = (idx / (W_out * H_out)) % {self.conv_config.kernel_size[1]};
    int k_h = (idx / (W_out * H_out * {self.conv_config.kernel_size[1]})) % {self.conv_config.kernel_size[0]};
    int c_in = (idx / (W_out * H_out * {self.conv_config.kernel_size[1]} * {self.conv_config.kernel_size[0]})) % C_in;
    int n = idx / (W_out * H_out * {self.conv_config.kernel_size[1]} * {self.conv_config.kernel_size[0]} * C_in);
    
    // 计算输入位置
    int h_in = h_out * {self.conv_config.stride[0]} - {self.conv_config.padding[0]} + k_h * {self.conv_config.dilation[0]};
    int w_in = w_out * {self.conv_config.stride[1]} - {self.conv_config.padding[1]} + k_w * {self.conv_config.dilation[1]};
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {{
        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
        val = input[input_idx];
    }}
    
    // 写入列数据
    int col_idx = ((((c_in * {self.conv_config.kernel_size[0]} + k_h) * {self.conv_config.kernel_size[1]} + k_w) * H_out + h_out) * W_out + w_out) * N + n;
    data_col[col_idx] = val;
}}

// GEMM卷积核函数
__global__ void {self.get_kernel_name()}_gemm(
    const float* weight,
    const float* data_col,
    const float* bias,
    float* output,
    int M, int N, int K
) {{
    // 简化的GEMM实现
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {{
        sum += weight[row * K + k] * data_col[k * N + col];
    }}
    
    {"if (bias != nullptr) sum += bias[row];" if self.conv_config.bias else ""}
    
    output[row * N + col] = sum;
}}
"""
    
    def _generate_cuda_backward(self) -> str:
        """生成CUDA反向传播代码"""
        return f"""
// CUDA反向传播实现 - 权重梯度
__global__ void {self.get_kernel_name()}_backward_weight(
    const float* input,
    const float* grad_output,
    float* grad_weight,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
) {{
    // 权重梯度计算实现
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = C_out * C_in * {self.conv_config.kernel_size[0]} * {self.conv_config.kernel_size[1]} / {self.conv_config.groups};
    
    if (idx >= total_weights) return;
    
    // 解析权重位置 (考虑分组)
    int group_size_in = C_in / {self.conv_config.groups};
    int group_size_out = C_out / {self.conv_config.groups};
    
    int local_idx = idx;
    int k_w = local_idx % {self.conv_config.kernel_size[1]};
    local_idx /= {self.conv_config.kernel_size[1]};
    int k_h = local_idx % {self.conv_config.kernel_size[0]};
    local_idx /= {self.conv_config.kernel_size[0]};
    int c_in_local = local_idx % group_size_in;
    local_idx /= group_size_in;
    int c_out = local_idx;
    
    int group_id = c_out / group_size_out;
    int c_in = group_id * group_size_in + c_in_local;
    
    float grad_sum = 0.0f;
    
    for (int n = 0; n < N; n++) {{
        for (int h_out = 0; h_out < H_out; h_out++) {{
            for (int w_out = 0; w_out < W_out; w_out++) {{
                int h_in = h_out * {self.conv_config.stride[0]} - {self.conv_config.padding[0]} + k_h * {self.conv_config.dilation[0]};
                int w_in = w_out * {self.conv_config.stride[1]} - {self.conv_config.padding[1]} + k_w * {self.conv_config.dilation[1]};
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {{
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    
                    grad_sum += input[input_idx] * grad_output[grad_output_idx];
                }}
            }}
        }}
    }}
    
    grad_weight[idx] = grad_sum;
}}

// CUDA反向传播实现 - 输入梯度  
__global__ void {self.get_kernel_name()}_backward_input(
    const float* weight,
    const float* grad_output,
    float* grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
) {{
    // 输入梯度计算实现
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = N * C_in * H_in * W_in;
    
    if (idx >= total_inputs) return;
    
    // 解析输入位置
    int w_in = idx % W_in;
    int h_in = (idx / W_in) % H_in;
    int c_in = (idx / (W_in * H_in)) % C_in;
    int n = idx / (W_in * H_in * C_in);
    
    float grad_sum = 0.0f;
    
    int group_size_in = C_in / {self.conv_config.groups};
    int group_size_out = C_out / {self.conv_config.groups};
    int group_id = c_in / group_size_in;
    int c_out_start = group_id * group_size_out;
    int c_out_end = c_out_start + group_size_out;
    
    for (int c_out = c_out_start; c_out < c_out_end; c_out++) {{
        for (int k_h = 0; k_h < {self.conv_config.kernel_size[0]}; k_h++) {{
            for (int k_w = 0; k_w < {self.conv_config.kernel_size[1]}; k_w++) {{
                int h_out = (h_in + {self.conv_config.padding[0]} - k_h * {self.conv_config.dilation[0]}) / {self.conv_config.stride[0]};
                int w_out = (w_in + {self.conv_config.padding[1]} - k_w * {self.conv_config.dilation[1]}) / {self.conv_config.stride[1]};
                
                bool valid_h = (h_in + {self.conv_config.padding[0]} - k_h * {self.conv_config.dilation[0]}) % {self.conv_config.stride[0]} == 0;
                bool valid_w = (w_in + {self.conv_config.padding[1]} - k_w * {self.conv_config.dilation[1]}) % {self.conv_config.stride[1]} == 0;
                
                if (valid_h && valid_w && h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {{
                    int weight_idx = ((c_out * group_size_in + (c_in % group_size_in)) * 
                                    {self.conv_config.kernel_size[0]} + k_h) * 
                                    {self.conv_config.kernel_size[1]} + k_w;
                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    
                    grad_sum += weight[weight_idx] * grad_output[grad_output_idx];
                }}
            }}
        }}
    }}
    
    grad_input[idx] = grad_sum;
}}
"""
    
    def _generate_cpu_forward(self) -> str:
        """生成CPU前向代码"""
        return f"""
// CPU卷积前向实现
void {self.get_kernel_name()}_forward_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out
) {{
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {{
        for (int c_out = 0; c_out < C_out; c_out++) {{
            for (int h_out = 0; h_out < H_out; h_out++) {{
                for (int w_out = 0; w_out < W_out; w_out++) {{
                    float sum = 0.0f;
                    
                    int group_size_in = C_in / {self.conv_config.groups};
                    int group_size_out = C_out / {self.conv_config.groups};
                    int group_id = c_out / group_size_out;
                    int c_in_start = group_id * group_size_in;
                    int c_in_end = c_in_start + group_size_in;
                    
                    for (int c_in = c_in_start; c_in < c_in_end; c_in++) {{
                        for (int k_h = 0; k_h < {self.conv_config.kernel_size[0]}; k_h++) {{
                            for (int k_w = 0; k_w < {self.conv_config.kernel_size[1]}; k_w++) {{
                                int h_in = h_out * {self.conv_config.stride[0]} - {self.conv_config.padding[0]} + k_h * {self.conv_config.dilation[0]};
                                int w_in = w_out * {self.conv_config.stride[1]} - {self.conv_config.padding[1]} + k_w * {self.conv_config.dilation[1]};
                                
                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {{
                                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                                    int weight_idx = ((c_out * group_size_in + (c_in - c_in_start)) * 
                                                    {self.conv_config.kernel_size[0]} + k_h) * 
                                                    {self.conv_config.kernel_size[1]} + k_w;
                                    
                                    sum += input[input_idx] * weight[weight_idx];
                                }}
                            }}
                        }}
                    }}
                    
                    {"// 添加偏置" if self.conv_config.bias else "// 无偏置"}
                    {"if (bias != nullptr) sum += bias[c_out];" if self.conv_config.bias else ""}
                    
                    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = sum;
                }}
            }}
        }}
    }}
}}
"""
    
    def _generate_cpu_backward(self) -> str:
        """生成CPU反向传播代码"""
        return "// CPU反向传播代码实现\n// 暂未实现，可参考CUDA版本进行适配"
    
    def get_performance_hints(self) -> List[str]:
        """获取性能优化建议"""
        hints = super().get_performance_hints()
        
        config = self.conv_config
        
        # 卷积特定的性能建议
        if config.groups == 1 and config.kernel_size == (3, 3) and not config.use_winograd:
            hints.append("考虑使用Winograd算法优化3x3卷积")
        
        if config.kernel_size[0] >= 7 or config.kernel_size[1] >= 7:
            hints.append("大卷积核建议使用FFT卷积算法")
        
        if config.groups > 1:
            hints.append("分组卷积已启用，可有效减少计算量")
        
        if not config.use_im2col and (config.kernel_size[0] > 3 or config.kernel_size[1] > 3):
            hints.append("较大卷积核建议使用im2col+GEMM算法")
        
        return hints