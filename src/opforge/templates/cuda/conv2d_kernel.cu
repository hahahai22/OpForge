/*
 * CUDA卷积算子实现 - 支持高级硬件指令
 * 自动生成 - 请勿手动修改
 */

#include "{{operator_type}}_kernel.h"
#include <stdio.h>
#include <assert.h>

{% if config.enable_tensor_core %}
#include <mma.h>
using namespace nvcuda;
{% endif %}

{% if config.enable_dot_instructions %}
// DOT指令宏定义
#define DOT2(a, b, c) asm("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c))
#define DOT4(a, b, c) asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c))
{% endif %}

{% if operator_type == "conv2d" %}

/**
 * CUDA 2D卷积前向内核实现
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
) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C_out * H_out * W_out;
    
    if (idx >= total_threads) return;
    
    // 解析输出位置
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);
    
    dtype_t sum = 0.0f;
    
    // 卷积计算
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int k_h = 0; k_h < K_h; k_h++) {
            for (int k_w = 0; k_w < K_w; k_w++) {
                // 计算输入位置
                int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                int w_in = w_out * stride_w - pad_w + k_w * dilation_w;
                
                // 边界检查
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    // 输入索引
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    // 权重索引  
                    int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // 添加偏置
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // 写入输出
    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}

/**
 * 优化版本 - 使用共享内存
 */
__global__ void {{kernel_name}}_forward_optimized(
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
) {
    // 共享内存声明
    extern __shared__ dtype_t shared_data[];
    
    // 线程和块索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // 计算输出位置
    int h_out = by;
    int w_out = bx * blockDim.x + tx;
    
    if (h_out >= H_out || w_out >= W_out) return;
    
    // 为每个批次和输出通道计算
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            dtype_t sum = 0.0f;
            
            // 卷积计算
            for (int c_in = 0; c_in < C_in; c_in++) {
                // 加载输入到共享内存
                // 这里可以添加更复杂的共享内存管理逻辑
                
                for (int k_h = 0; k_h < K_h; k_h++) {
                    for (int k_w = 0; k_w < K_w; k_w++) {
                        int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                        int w_in = w_out * stride_w - pad_w + k_w * dilation_w;
                        
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                            int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // 添加偏置
            if (bias != nullptr) {
                sum += bias[c_out];
            }
            
            // 写入输出
            int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
            output[output_idx] = sum;
        }
    }
}

/**
 * 权重梯度计算内核
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = C_out * C_in * K_h * K_w;
    
    if (idx >= total_weights) return;
    
    // 解析权重位置
    int k_w = idx % K_w;
    int k_h = (idx / K_w) % K_h;
    int c_in = (idx / (K_w * K_h)) % C_in;
    int c_out = idx / (K_w * K_h * C_in);
    
    dtype_t grad_sum = 0.0f;
    
    // 遍历所有批次和输出位置
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                // 计算对应的输入位置
                int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                int w_in = w_out * stride_w - pad_w + k_w * dilation_w;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    
                    grad_sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }
    
    grad_weight[idx] = grad_sum;
}

/**
 * 输入梯度计算内核
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = N * C_in * H_in * W_in;
    
    if (idx >= total_inputs) return;
    
    // 解析输入位置
    int w_in = idx % W_in;
    int h_in = (idx / W_in) % H_in;
    int c_in = (idx / (W_in * H_in)) % C_in;
    int n = idx / (W_in * H_in * C_in);
    
    dtype_t grad_sum = 0.0f;
    
    // 遍历所有输出通道和卷积核位置
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int k_h = 0; k_h < K_h; k_h++) {
            for (int k_w = 0; k_w < K_w; k_w++) {
                // 计算对应的输出位置
                int h_out = (h_in + pad_h - k_h * dilation_h) / stride_h;
                int w_out = (w_in + pad_w - k_w * dilation_w) / stride_w;
                
                // 检查是否为有效的输出位置
                bool valid_h = (h_in + pad_h - k_h * dilation_h) % stride_h == 0;
                bool valid_w = (w_in + pad_w - k_w * dilation_w) % stride_w == 0;
                
                if (valid_h && valid_w && h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                    int grad_output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    
                    grad_sum += weight[weight_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }
    
    grad_input[idx] = grad_sum;
}

{% endif %}

/**
 * 启动器函数实现
 */
extern "C" 
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
    cudaStream_t stream
) {
    // 计算网格和块配置
    {% if operator_type == "conv2d" %}
    int total_threads = {{parameters.get('batch_size', 'N')}} * {{parameters.get('out_channels', 'C_out')}} * {{parameters.get('out_height', 'H_out')}} * {{parameters.get('out_width', 'W_out')}};
    {% else %}
    int total_threads = {{grid_size * block_size}};
    {% endif %}
    
    dim3 block_size({{block_size}});
    dim3 grid_size((total_threads + {{block_size}} - 1) / {{block_size}});
    
{% if optimization_level >= 1 %}
    // 使用优化版本
    {{kernel_name}}_forward_optimized<<<grid_size, block_size, {{shared_memory_size}}, stream>>>(
{% else %}
    // 使用基础版本
    {{kernel_name}}_forward<<<grid_size, block_size, 0, stream>>>(
{% endif %}
        input, weight, bias, output,
        {% for shape in input_shapes %}
        {% for dim in shape.dims %}{{dim}}{% if not loop.last %}, {% endif %}{% endfor %}{% if not loop.last %},{% endif %}
        {% endfor %},
        {% for param_name, param_value in parameters.items() %}
        {{param_name}}{% if not loop.last %}, {% endif %}
        {% endfor %}
    );
    
{% if debug_mode %}
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // 同步等待
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return err;
    }
{% endif %}
    
    return cudaSuccess;
}