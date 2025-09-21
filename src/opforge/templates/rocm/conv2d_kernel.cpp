/*
 * ROCm HIP 2D卷积内核实现
 * 自动生成 - 请勿手动修改
 */

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <miopen/miopen.h>

// 数据类型定义
typedef {{dtype|dtype_to_c}} dtype_t;

{% if config.enable_buffer_ops %}
// 缓冲区操作宏
#define BUFFER_LOAD_DWORD(dst, src, offset) \
    dst = __builtin_amdgcn_raw_buffer_load_i32(src, offset, 0, 0)

#define BUFFER_STORE_DWORD(dst, src, offset) \
    __builtin_amdgcn_raw_buffer_store_i32(src, dst, offset, 0, 0)
{% endif %}

/**
 * ROCm HIP 2D卷积前向内核 - 直接实现
 */
__global__ void {{kernel_name}}_hip_direct(
    const dtype_t* __restrict__ input,
    const dtype_t* __restrict__ weight,
    const dtype_t* __restrict__ bias,
    dtype_t* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    // 使用ROCm的线程和块索引
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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
                int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                int w_in = w_out * stride_w - pad_w + k_w * dilation_w;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                    
                    {% if config.enable_buffer_ops %}
                    // 使用缓冲区加载指令
                    dtype_t input_val, weight_val;
                    BUFFER_LOAD_DWORD(input_val, input, input_idx * sizeof(dtype_t));
                    BUFFER_LOAD_DWORD(weight_val, weight, weight_idx * sizeof(dtype_t));
                    sum += input_val * weight_val;
                    {% else %}
                    sum += input[input_idx] * weight[weight_idx];
                    {% endif %}
                }
            }
        }
    }
    
    // 添加偏置
    {% if parameters.use_bias %}
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    {% endif %}
    
    // 写入输出
    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[output_idx] = sum;
}

{% if config.hardware_config and 'MFMA' in config.hardware_config.supported_instructions %}
/**
 * ROCm MFMA优化的2D卷积内核
 * 使用Matrix Fused Multiply Add指令
 */
__global__ void {{kernel_name}}_hip_mfma(
    const dtype_t* __restrict__ input,
    const dtype_t* __restrict__ weight,
    const dtype_t* __restrict__ bias,
    dtype_t* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    // 使用共享内存进行数据缓存
    __shared__ dtype_t shared_input[16 * 16];
    __shared__ dtype_t shared_weight[16 * 16];
    
    int tid = hipThreadIdx_x;
    int bid = hipBlockIdx_x;
    
    // MFMA矩阵尺寸 (16x16x16)
    const int MFMA_M = 16;
    const int MFMA_N = 16;
    const int MFMA_K = 16;
    
    // 计算当前块处理的输出位置
    int block_c_out = bid * MFMA_M;
    int block_hw = hipBlockIdx_y * MFMA_N;
    
    if (block_c_out >= C_out || block_hw >= H_out * W_out) return;
    
    // 累加器初始化
    float acc[MFMA_M] = {0.0f};
    
    // 卷积计算循环
    for (int c_in = 0; c_in < C_in; c_in += MFMA_K) {
        // 加载输入数据到共享内存
        if (tid < MFMA_N * MFMA_K) {
            int hw_idx = (block_hw + tid / MFMA_K) % (H_out * W_out);
            int h_out = hw_idx / W_out;
            int w_out = hw_idx % W_out;
            int c_idx = c_in + tid % MFMA_K;
            
            // 计算对应的输入位置（这里简化了卷积映射）
            if (c_idx < C_in) {
                int input_idx = c_idx * H_in * W_in + h_out * W_in + w_out;
                shared_input[tid] = (input_idx < N * C_in * H_in * W_in) ? input[input_idx] : 0.0f;
            } else {
                shared_input[tid] = 0.0f;
            }
        }
        
        // 加载权重数据到共享内存
        if (tid < MFMA_M * MFMA_K) {
            int c_out_idx = block_c_out + tid / MFMA_K;
            int c_in_idx = c_in + tid % MFMA_K;
            
            if (c_out_idx < C_out && c_in_idx < C_in) {
                // 简化的权重索引计算
                int weight_idx = c_out_idx * C_in + c_in_idx;
                shared_weight[tid] = weight[weight_idx];
            } else {
                shared_weight[tid] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // 使用MFMA指令进行矩阵乘法累加
        // 注意：这里是伪代码，实际的MFMA指令需要内联汇编
        #ifdef __gfx908__
        // GFX908架构的MFMA指令
        for (int i = 0; i < MFMA_M; i += 4) {
            // v_mfma_f32_16x16x16f16 指令
            asm volatile (
                "v_mfma_f32_16x16x16f16 %0, %1, %2, %0"
                : "+v"(acc[i])
                : "v"(shared_input[i]), "v"(shared_weight[i])
            );
        }
        #endif
        
        __syncthreads();
    }
    
    // 写入输出结果
    if (tid < MFMA_M) {
        int c_out_idx = block_c_out + tid;
        if (c_out_idx < C_out) {
            int hw_idx = block_hw;
            if (hw_idx < H_out * W_out) {
                int output_idx = c_out_idx * H_out * W_out + hw_idx;
                
                {% if parameters.use_bias %}
                if (bias != nullptr) {
                    acc[tid] += bias[c_out_idx];
                }
                {% endif %}
                
                output[output_idx] = acc[tid];
            }
        }
    }
}
{% endif %}

/**
 * ROCm HIP卷积启动器函数
 */
extern "C"
hipError_t launch_{{kernel_name}}_hip(
    const dtype_t* input,
    const dtype_t* weight,
    const dtype_t* bias,
    dtype_t* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    hipStream_t stream = nullptr
) {
    // 计算网格和块配置
    int total_threads = N * C_out * H_out * W_out;
    
    {% if config.optimization_level >= 2 %}
    dim3 block_size(256);
    {% elif config.optimization_level == 1 %}
    dim3 block_size(128);
    {% else %}
    dim3 block_size(64);
    {% endif %}
    
    dim3 grid_size((total_threads + block_size.x - 1) / block_size.x);
    
    {% if config.hardware_config and 'MFMA' in config.hardware_config.supported_instructions %}
    // 如果支持MFMA，使用优化版本
    if (C_out >= 16 && H_out * W_out >= 16) {
        dim3 mfma_grid((C_out + 15) / 16, (H_out * W_out + 15) / 16);
        dim3 mfma_block(64);  // 每个MFMA需要64个线程
        
        hipLaunchKernelGGL({{kernel_name}}_hip_mfma, mfma_grid, mfma_block, 
                          16 * 16 * 2 * sizeof(dtype_t), stream,
                          input, weight, bias, output,
                          N, C_in, H_in, W_in, C_out, H_out, W_out,
                          K_h, K_w, stride_h, stride_w, pad_h, pad_w,
                          dilation_h, dilation_w);
    } else
    {% endif %}
    {
        // 使用标准直接卷积
        hipLaunchKernelGGL({{kernel_name}}_hip_direct, grid_size, block_size, 0, stream,
                          input, weight, bias, output,
                          N, C_in, H_in, W_in, C_out, H_out, W_out,
                          K_h, K_w, stride_h, stride_w, pad_h, pad_w,
                          dilation_h, dilation_w);
    }
    
    {% if debug_mode %}
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel launch failed: %s\n", hipGetErrorString(err));
        return err;
    }
    
    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        printf("Kernel execution failed: %s\n", hipGetErrorString(err));
        return err;
    }
    {% endif %}
    
    return hipSuccess;
}

// ROCm性能配置
#define ROCM_BLOCK_SIZE {{block_size}}
#define ROCM_GRID_SIZE {{grid_size}}

{% if optimization_level >= 1 %}
#define USE_SHARED_MEMORY
{% endif %}

{% if optimization_level >= 2 %}
#define USE_MFMA_INSTRUCTIONS
#define AGGRESSIVE_UNROLL
{% endif %}

{% if debug_mode %}
#define DEBUG_MODE
#define CHECK_HIP_ERROR(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    } \
} while(0)
{% else %}
#define CHECK_HIP_ERROR(call) call
{% endif %}