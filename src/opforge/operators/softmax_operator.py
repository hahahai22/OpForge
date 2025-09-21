"""
Softmax算子生成器

支持生成数值稳定的高性能Softmax算子，包括标准Softmax、带温度的Softmax、
以及大规模序列的优化版本。
"""

from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import math

from ..core.operator_base import OperatorBase, OperatorConfig, TensorShape, DataType, Backend


@dataclass  
class SoftmaxConfig(OperatorConfig):
    """Softmax配置"""
    # 基本参数
    dim: int = -1  # 计算softmax的维度
    temperature: float = 1.0  # 温度参数
    
    # 数值稳定性选项
    use_log_softmax: bool = False  # 是否计算log_softmax
    eps: float = 1e-8  # 数值稳定性epsilon
    
    # 性能优化选项
    use_online_softmax: bool = True  # 在线softmax算法
    use_block_reduce: bool = True   # 块级归约
    use_warp_reduce: bool = True    # warp级归约
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.temperature <= 0:
            raise ValueError("温度参数必须大于0")
        if self.eps <= 0:
            raise ValueError("epsilon必须大于0")


class SoftmaxOperator(OperatorBase):
    """Softmax算子生成器"""
    
    def __init__(self, config: SoftmaxConfig):
        super().__init__(config)
        self.softmax_config = config
        
        # 设置参数
        self.add_parameter("dim", config.dim)
        self.add_parameter("temperature", config.temperature)
        self.add_parameter("use_log_softmax", config.use_log_softmax)
        self.add_parameter("eps", config.eps)
    
    def get_operator_type(self) -> str:
        """返回算子类型"""
        if self.softmax_config.use_log_softmax:
            return "log_softmax"
        else:
            return "softmax"
    
    def validate_config(self) -> bool:
        """验证配置"""
        config = self.softmax_config
        
        if config.temperature <= 0:
            return False
        if config.eps <= 0:
            return False
            
        return True
    
    def infer_output_shape(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """推断输出形状"""
        if not input_shapes:
            raise ValueError("需要至少一个输入张量")
        
        input_shape = input_shapes[0]
        
        # 验证维度参数
        dim = self.softmax_config.dim
        if dim < 0:
            dim = len(input_shape.dims) + dim
        
        if dim < 0 or dim >= len(input_shape.dims):
            raise ValueError(f"维度参数 {self.softmax_config.dim} 超出张量维度范围")
        
        # Softmax输出形状与输入相同
        output_shape = TensorShape(
            dims=input_shape.dims.copy(),
            dtype=input_shape.dtype,
            name="softmax_output"
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
        config = self.softmax_config
        
        if config.use_online_softmax:
            return self._generate_cuda_online_softmax()
        else:
            return self._generate_cuda_standard_softmax()
    
    def _generate_cuda_online_softmax(self) -> str:
        """生成CUDA在线Softmax代码"""
        config = self.softmax_config
        log_suffix = "_log" if config.use_log_softmax else ""
        
        return f"""
// CUDA在线Softmax算法实现
// 使用单次扫描算法，数值稳定且内存高效

__device__ float warp_reduce_max(float val) {{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }}
    return val;
}}

__device__ float warp_reduce_sum(float val) {{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        val += __shfl_down_sync(0xffffffff, val, offset);
    }}
    return val;
}}

__global__ void {self.get_kernel_name()}_forward{log_suffix}(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    
    // 共享内存用于存储中间结果
    __shared__ float shared_max[32];  // 假设最多32个warp
    __shared__ float shared_sum[32];
    
    const float* input_row = input + (batch_idx * seq_len + seq_idx) * dim_size;
    float* output_row = output + (batch_idx * seq_len + seq_idx) * dim_size;
    
    // 第一步：在线计算最大值
    float thread_max = -INFINITY;
    for (int i = tid; i < dim_size; i += blockDim.x) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        thread_max = fmaxf(thread_max, val);
    }}
    
    // Warp级归约求最大值
    {"thread_max = warp_reduce_max(thread_max);" if config.use_warp_reduce else ""}
    
    // 存储到共享内存
    if (lane_id == 0) {{
        shared_max[warp_id] = thread_max;
    }}
    __syncthreads();
    
    // 块级归约求全局最大值
    float global_max = -INFINITY;
    {"if (tid < 32) {" if config.use_block_reduce else "if (tid == 0) {"}
        {"for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++) {" if config.use_block_reduce else "for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++) {"}
            global_max = fmaxf(global_max, shared_max[i]);
        }}
        {"global_max = warp_reduce_max(global_max);" if config.use_block_reduce and config.use_warp_reduce else ""}
    }}
    {"if (tid == 0) shared_max[0] = global_max;" if config.use_block_reduce else "shared_max[0] = global_max;"}
    __syncthreads();
    global_max = shared_max[0];
    
    // 第二步：计算指数和的累加
    float thread_sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        float exp_val = expf(val - global_max);
        thread_sum += exp_val;
        
        // 暂存指数值，后续需要
        output_row[i] = exp_val;
    }}
    
    // Warp级归约求和
    {"thread_sum = warp_reduce_sum(thread_sum);" if config.use_warp_reduce else ""}
    
    // 存储到共享内存
    if (lane_id == 0) {{
        shared_sum[warp_id] = thread_sum;
    }}
    __syncthreads();
    
    // 块级归约求全局和
    float global_sum = 0.0f;
    {"if (tid < 32) {" if config.use_block_reduce else "if (tid == 0) {"}
        {"for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++) {" if config.use_block_reduce else "for (int i = 0; i < (blockDim.x + warpSize - 1) / warpSize; i++) {"}
            global_sum += shared_sum[i];
        }}
        {"global_sum = warp_reduce_sum(global_sum);" if config.use_block_reduce and config.use_warp_reduce else ""}
    }}
    {"if (tid == 0) shared_sum[0] = global_sum;" if config.use_block_reduce else "shared_sum[0] = global_sum;"}
    __syncthreads();
    global_sum = shared_sum[0];
    
    // 第三步：归一化
    global_sum = fmaxf(global_sum, {config.eps}f);  // 数值稳定性
    
    for (int i = tid; i < dim_size; i += blockDim.x) {{
        {"output_row[i] = logf(output_row[i] + " + str(config.eps) + "f) - logf(global_sum);" if config.use_log_softmax else "output_row[i] = output_row[i] / global_sum;"}
    }}
}}

// 快速Softmax版本 (适用于小序列长度)
__global__ void {self.get_kernel_name()}_forward_fast{log_suffix}(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const float* input_row = input + idx * dim_size;
    float* output_row = output + idx * dim_size;
    
    // 计算最大值
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; i++) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        max_val = fmaxf(max_val, val);
    }}
    
    // 计算指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        float exp_val = expf(val - max_val);
        output_row[i] = exp_val;
        sum_exp += exp_val;
    }}
    
    // 归一化
    sum_exp = fmaxf(sum_exp, {config.eps}f);
    for (int i = 0; i < dim_size; i++) {{
        {"output_row[i] = logf(output_row[i] + " + str(config.eps) + "f) - logf(sum_exp);" if config.use_log_softmax else "output_row[i] = output_row[i] / sum_exp;"}
    }}
}}
"""
    
    def _generate_cuda_standard_softmax(self) -> str:
        """生成CUDA标准Softmax代码"""
        config = self.softmax_config
        log_suffix = "_log" if config.use_log_softmax else ""
        
        return f"""
// CUDA标准Softmax实现（三遍扫描）
__global__ void {self.get_kernel_name()}_forward{log_suffix}(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const float* input_row = input + idx * dim_size;
    float* output_row = output + idx * dim_size;
    
    // 第一遍：找最大值
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; i++) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        max_val = fmaxf(max_val, val);
    }}
    
    // 第二遍：计算指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        float exp_val = expf(val - max_val);
        sum_exp += exp_val;
    }}
    
    // 第三遍：归一化
    sum_exp = fmaxf(sum_exp, {config.eps}f);
    for (int i = 0; i < dim_size; i++) {{
        float val = input_row[i];
        {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
        float exp_val = expf(val - max_val);
        {"output_row[i] = logf(exp_val + " + str(config.eps) + "f) - logf(sum_exp);" if config.use_log_softmax else "output_row[i] = exp_val / sum_exp;"}
    }}
}}
"""
    
    def _generate_cuda_backward(self) -> str:
        """生成CUDA反向传播代码"""
        config = self.softmax_config
        
        if config.use_log_softmax:
            return self._generate_cuda_log_softmax_backward()
        else:
            return self._generate_cuda_softmax_backward()
    
    def _generate_cuda_softmax_backward(self) -> str:
        """生成CUDA Softmax反向传播代码"""
        config = self.softmax_config
        
        return f"""
// CUDA Softmax反向传播实现
__global__ void {self.get_kernel_name()}_backward(
    const float* grad_output,
    const float* softmax_output,
    float* grad_input,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const float* grad_out_row = grad_output + idx * dim_size;
    const float* softmax_row = softmax_output + idx * dim_size;
    float* grad_in_row = grad_input + idx * dim_size;
    
    // 计算 sum(grad_output * softmax_output)
    float sum_grad_softmax = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        sum_grad_softmax += grad_out_row[i] * softmax_row[i];
    }}
    
    // 计算梯度: grad_input = softmax * (grad_output - sum_grad_softmax)
    for (int i = 0; i < dim_size; i++) {{
        grad_in_row[i] = softmax_row[i] * (grad_out_row[i] - sum_grad_softmax);
        {"grad_in_row[i] /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
    }}
}}
"""
    
    def _generate_cuda_log_softmax_backward(self) -> str:
        """生成CUDA LogSoftmax反向传播代码"""
        config = self.softmax_config
        
        return f"""
// CUDA LogSoftmax反向传播实现
__global__ void {self.get_kernel_name()}_backward(
    const float* grad_output,
    const float* log_softmax_output,
    float* grad_input,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (idx >= total_sequences) return;
    
    const float* grad_out_row = grad_output + idx * dim_size;
    const float* log_softmax_row = log_softmax_output + idx * dim_size;
    float* grad_in_row = grad_input + idx * dim_size;
    
    // 计算 sum(grad_output)
    float sum_grad = 0.0f;
    for (int i = 0; i < dim_size; i++) {{
        sum_grad += grad_out_row[i];
    }}
    
    // 计算梯度: grad_input = grad_output - exp(log_softmax) * sum_grad
    for (int i = 0; i < dim_size; i++) {{
        float softmax_val = expf(log_softmax_row[i]);
        grad_in_row[i] = grad_out_row[i] - softmax_val * sum_grad;
        {"grad_in_row[i] /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
    }}
}}
"""
    
    def _generate_cpu_forward(self) -> str:
        """生成CPU前向代码"""
        config = self.softmax_config
        log_suffix = "_log" if config.use_log_softmax else ""
        
        return f"""
// CPU Softmax前向实现
void {self.get_kernel_name()}_forward{log_suffix}_cpu(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
        for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {{
            const float* input_row = input + (batch_idx * seq_len + seq_idx) * dim_size;
            float* output_row = output + (batch_idx * seq_len + seq_idx) * dim_size;
            
            // 找最大值
            float max_val = -INFINITY;
            for (int i = 0; i < dim_size; i++) {{
                float val = input_row[i];
                {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
                max_val = fmaxf(max_val, val);
            }}
            
            // 计算指数和
            float sum_exp = 0.0f;
            for (int i = 0; i < dim_size; i++) {{
                float val = input_row[i];
                {"val /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
                float exp_val = expf(val - max_val);
                output_row[i] = exp_val;
                sum_exp += exp_val;
            }}
            
            // 归一化
            sum_exp = fmaxf(sum_exp, {config.eps}f);
            for (int i = 0; i < dim_size; i++) {{
                {"output_row[i] = logf(output_row[i] + " + str(config.eps) + "f) - logf(sum_exp);" if config.use_log_softmax else "output_row[i] = output_row[i] / sum_exp;"}
            }}
        }}
    }}
}}
"""
    
    def _generate_cpu_backward(self) -> str:
        """生成CPU反向传播代码"""
        config = self.softmax_config
        
        if config.use_log_softmax:
            grad_formula = "grad_out_row[i] - expf(output_row[i]) * sum_grad"
        else:
            grad_formula = "output_row[i] * (grad_out_row[i] - sum_grad_output)"
        
        return f"""
// CPU Softmax反向传播实现
void {self.get_kernel_name()}_backward_cpu(
    const float* grad_output,
    const float* output,
    float* grad_input,
    int batch_size,
    int seq_len,
    int dim_size
) {{
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
        for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {{
            const float* grad_out_row = grad_output + (batch_idx * seq_len + seq_idx) * dim_size;
            const float* output_row = output + (batch_idx * seq_len + seq_idx) * dim_size;
            float* grad_in_row = grad_input + (batch_idx * seq_len + seq_idx) * dim_size;
            
            {"// 计算 sum(grad_output)" if config.use_log_softmax else "// 计算 sum(grad_output * output)"}
            float sum_grad{"" if config.use_log_softmax else "_output"} = 0.0f;
            for (int i = 0; i < dim_size; i++) {{
                {"sum_grad += grad_out_row[i];" if config.use_log_softmax else "sum_grad_output += grad_out_row[i] * output_row[i];"}
            }}
            
            // 计算梯度
            for (int i = 0; i < dim_size; i++) {{
                grad_in_row[i] = {grad_formula};
                {"grad_in_row[i] /= " + str(config.temperature) + ";" if config.temperature != 1.0 else ""}
            }}
        }}
    }}
}}
"""
    
    def get_performance_hints(self) -> List[str]:
        """获取性能优化建议"""
        hints = super().get_performance_hints()
        
        config = self.softmax_config
        
        # Softmax特定的性能建议
        if not config.use_online_softmax:
            hints.append("建议使用在线Softmax算法以提高内存效率")
        
        if config.temperature != 1.0:
            hints.append("使用了温度缩放，可能影响数值精度")
        
        if config.use_log_softmax:
            hints.append("使用LogSoftmax可以提高数值稳定性")
        
        # 根据维度大小给出建议
        for shape in self.input_shapes:
            if len(shape.dims) > 0:
                last_dim = shape.dims[-1]
                if isinstance(last_dim, int):
                    if last_dim > 10000:
                        hints.append("序列长度较大，建议使用块级归约优化")
                    elif last_dim < 32:
                        hints.append("序列长度较小，可以使用快速版本")
        
        return hints