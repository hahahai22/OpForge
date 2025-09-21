"""
Triton 2D卷积内核实现
自动生成 - 请勿手动修改
"""

import triton
import triton.language as tl
import torch


@triton.jit
def {{kernel_name}}_triton_kernel(
    # 输入张量指针
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    
    # 张量形状参数
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    
    # 卷积参数
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    
    # 步长信息
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_out_stride, weight_in_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    
    # 块大小
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    
    # 硬件指令配置
    {% if config.enable_tensor_core %}
    USE_TENSOR_CORE: tl.constexpr,
    {% endif %}
    {% if config.enable_dot_instructions %}
    USE_DOT_PRODUCT: tl.constexpr,
    {% endif %}
):
    """
    Triton实现的2D卷积内核
    
    使用块级tiling和高效的内存访问模式
    支持Tensor Core和DOT指令优化
    """
    
    # 获取程序ID
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # 计算输出位置
    batch_idx = pid_batch
    out_c_start = pid_out_c * BLOCK_SIZE_C
    h_start = pid_h * BLOCK_SIZE_H
    
    # 边界检查
    if batch_idx >= N:
        return
    if out_c_start >= C_out:
        return
    if h_start >= H_out:
        return
    
    # 创建输出通道掩码
    out_c_offsets = out_c_start + tl.arange(0, BLOCK_SIZE_C)
    out_c_mask = out_c_offsets < C_out
    
    # 对每个输出位置进行计算
    for w_out in range(0, W_out, BLOCK_SIZE_W):
        w_offsets = w_out + tl.arange(0, BLOCK_SIZE_W)
        w_mask = w_offsets < W_out
        
        # 初始化累加器
        {% if config.enable_tensor_core %}
        # 使用Tensor Core优化的累加器
        acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        {% else %}
        acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        {% endif %}
        
        # 卷积计算循环
        for c_in in range(C_in):
            for k_h in range(K_h):
                for k_w in range(K_w):
                    # 计算输入位置
                    h_in_base = (h_start + tl.arange(0, BLOCK_SIZE_H)[:, None]) * stride_h - pad_h + k_h * dilation_h
                    w_in_base = w_offsets[None, :] * stride_w - pad_w + k_w * dilation_w
                    
                    # 边界检查
                    h_valid = (h_in_base >= 0) & (h_in_base < H_in)
                    w_valid = (w_in_base >= 0) & (w_in_base < W_in)
                    valid_mask = h_valid & w_valid & w_mask[None, :]
                    
                    # 加载输入数据
                    input_offsets = (batch_idx * input_batch_stride + 
                                   c_in * input_channel_stride +
                                   h_in_base * input_h_stride +
                                   w_in_base * input_w_stride)
                    
                    input_data = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
                    
                    # 加载权重数据
                    weight_offsets = (out_c_offsets[:, None, None] * weight_out_stride +
                                    c_in * weight_in_stride +
                                    k_h * weight_h_stride +
                                    k_w * weight_w_stride)
                    
                    weight_data = tl.load(weight_ptr + weight_offsets, mask=out_c_mask[:, None, None])
                    
                    {% if config.enable_dot_instructions %}
                    # 使用DOT指令优化的矩阵乘法
                    acc += tl.dot(weight_data, input_data, allow_tf32={{config.optimization_level >= 2}})
                    {% else %}
                    # 标准乘加操作
                    acc += weight_data * input_data[None, :, :]
                    {% endif %}
        
        # 添加偏置
        {% if parameters.use_bias %}
        if bias_ptr is not None:
            bias_offsets = out_c_offsets
            bias_data = tl.load(bias_ptr + bias_offsets, mask=out_c_mask)
            acc += bias_data[:, None, None]
        {% endif %}
        
        # 存储输出
        for h_idx in range(BLOCK_SIZE_H):
            if h_start + h_idx < H_out:
                output_offsets = (batch_idx * output_batch_stride +
                                out_c_offsets[:, None] * output_channel_stride +
                                (h_start + h_idx) * output_h_stride +
                                w_offsets[None, :] * output_w_stride)
                
                output_mask = out_c_mask[:, None] & w_mask[None, :]
                tl.store(output_ptr + output_offsets, acc[:, h_idx, :], mask=output_mask)


def {{kernel_name}}_triton_launcher(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1)
) -> torch.Tensor:
    """
    Triton卷积启动器函数
    
    Args:
        input_tensor: 输入张量 [N, C_in, H_in, W_in]
        weight_tensor: 权重张量 [C_out, C_in, K_h, K_w]
        bias_tensor: 偏置张量 [C_out] (可选)
        stride: 步长 (stride_h, stride_w)
        padding: 填充 (pad_h, pad_w)
        dilation: 扩张 (dilation_h, dilation_w)
    
    Returns:
        输出张量 [N, C_out, H_out, W_out]
    """
    
    # 获取张量维度
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, K_h, K_w = weight_tensor.shape
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    # 计算输出尺寸
    H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    
    # 创建输出张量
    output_tensor = torch.empty((N, C_out, H_out, W_out), 
                               dtype=input_tensor.dtype, 
                               device=input_tensor.device)
    
    # 计算步长
    input_strides = input_tensor.stride()
    weight_strides = weight_tensor.stride()
    output_strides = output_tensor.stride()
    
    # 确定块大小
    {% if config.optimization_level >= 2 %}
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    {% elif config.optimization_level == 1 %}
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    {% else %}
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    {% endif %}
    
    # 计算网格大小
    grid_batch = N
    grid_out_c = triton.cdiv(C_out, BLOCK_SIZE_C)
    grid_h = triton.cdiv(H_out, BLOCK_SIZE_H)
    
    # 启动内核
    {{kernel_name}}_triton_kernel[(grid_batch, grid_out_c, grid_h)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output_tensor,
        
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        
        *input_strides,
        *weight_strides,
        *output_strides,
        
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        
        {% if config.enable_tensor_core %}
        USE_TENSOR_CORE=True,
        {% endif %}
        {% if config.enable_dot_instructions %}
        USE_DOT_PRODUCT=True,
        {% endif %}
    )
    
    return output_tensor


# 性能配置
TRITON_CONFIG = {
    'BLOCK_SIZE_C': [32, 64, 128],
    'BLOCK_SIZE_H': [16, 32, 64],
    'BLOCK_SIZE_W': [16, 32, 64],
    'num_stages': [1, 2, 4],
    'num_warps': [4, 8, 16],
}

{% if config.optimization_level >= 2 %}
# 自动调优配置
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': c, 'BLOCK_SIZE_H': h, 'BLOCK_SIZE_W': w, 
                      'num_stages': s, 'num_warps': n},
                     num_stages=s, num_warps=n)
        for c in TRITON_CONFIG['BLOCK_SIZE_C']
        for h in TRITON_CONFIG['BLOCK_SIZE_H'] 
        for w in TRITON_CONFIG['BLOCK_SIZE_W']
        for s in TRITON_CONFIG['num_stages']
        for n in TRITON_CONFIG['num_warps']
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'C_out', 'H_out', 'W_out'],
)
{% endif %}
def {{kernel_name}}_optimized(input_tensor, weight_tensor, bias_tensor=None, **kwargs):
    """自动调优版本的卷积函数"""
    return {{kernel_name}}_triton_launcher(input_tensor, weight_tensor, bias_tensor, **kwargs)