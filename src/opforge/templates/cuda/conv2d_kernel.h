/*
 * CUDA卷积算子头文件
 * 自动生成 - 请勿手动修改
 */

#ifndef {{operator_type.upper()}}_CUDA_KERNEL_H
#define {{operator_type.upper()}}_CUDA_KERNEL_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

// 数据类型定义
typedef {{dtype|dtype_to_c}} dtype_t;

// 内核函数声明
extern "C" {

{% if operator_type == "conv2d" %}
/**
 * CUDA 2D卷积前向内核
 * @param input: 输入张量 [N, C_in, H_in, W_in]
 * @param weight: 权重张量 [C_out, C_in, K_h, K_w]  
 * @param bias: 偏置张量 [C_out] (可选)
 * @param output: 输出张量 [N, C_out, H_out, W_out]
 * @param N: 批次大小
 * @param C_in: 输入通道数
 * @param H_in: 输入高度
 * @param W_in: 输入宽度
 * @param C_out: 输出通道数
 * @param H_out: 输出高度
 * @param W_out: 输出宽度
 * @param K_h: 卷积核高度
 * @param K_w: 卷积核宽度
 * @param stride_h: 高度方向步长
 * @param stride_w: 宽度方向步长
 * @param pad_h: 高度方向填充
 * @param pad_w: 宽度方向填充
 * @param dilation_h: 高度方向扩张
 * @param dilation_w: 宽度方向扩张
 */
__global__ void {{kernel_name}}_forward(
    const dtype_t* input,
    const dtype_t* weight,
    const dtype_t* bias,
    dtype_t* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
);

/**
 * CUDA 2D卷积反向内核 (权重梯度)
 */
__global__ void {{kernel_name}}_backward_weight(
    const dtype_t* input,
    const dtype_t* grad_output,
    dtype_t* grad_weight,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
);

/**
 * CUDA 2D卷积反向内核 (输入梯度)
 */
__global__ void {{kernel_name}}_backward_input(
    const dtype_t* weight,
    const dtype_t* grad_output,
    dtype_t* grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
);

{% endif %}

// 启动器函数
cudaError_t launch_{{kernel_name}}_forward(
    const dtype_t* input,
    const dtype_t* weight, 
    const dtype_t* bias,
    dtype_t* output,
    {% for shape in input_shapes %}
    {% for dim in shape.dims %}int {{dim}}{% if not loop.last %}, {% endif %}{% endfor %}{% if not loop.last %},{% endif %}
    {% endfor %},
    {% for param_name, param_value in parameters.items() %}
    int {{param_name}}{% if not loop.last %}, {% endif %}
    {% endfor %},
    cudaStream_t stream = nullptr
);

}  // extern "C"

// 性能配置
#define BLOCK_SIZE_X {{block_size}}
#define BLOCK_SIZE_Y 1
#define GRID_SIZE_X {{grid_size}}
#define GRID_SIZE_Y 1

// 共享内存配置
#define SHARED_MEM_SIZE {{shared_memory_size}}

// 优化标志
{% if optimization_level >= 1 %}
#define USE_TENSOR_CORES
{% endif %}

{% if optimization_level >= 2 %}
#define USE_FAST_MATH
#define AGGRESSIVE_UNROLL
{% endif %}

{% if debug_mode %}
#define DEBUG_MODE
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)
{% else %}
#define CHECK_CUDA_ERROR(call) call
{% endif %}

#endif // {{operator_type.upper()}}_CUDA_KERNEL_H