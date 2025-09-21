/*
 * CPU卷积算子实现
 * 自动生成 - 请勿手动修改
 */

#include <omp.h>
#include <cblas.h>
#include <immintrin.h>  // AVX指令

// 数据类型定义
typedef {{dtype|dtype_to_c}} dtype_t;

/**
 * CPU 2D卷积前向实现 - OpenMP并行版本
 */
void {{kernel_name}}_forward_cpu(
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
    // OpenMP并行化
    #pragma omp parallel for collapse(4) schedule(dynamic)
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
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
                                    
                                    sum += input[input_idx] * weight[weight_idx];
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
                    
                    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

{% if optimization_level >= 1 %}
/**
 * CPU 2D卷积前向实现 - AVX向量化版本
 */
void {{kernel_name}}_forward_cpu_vectorized(
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
    const int vec_size = 8;  // AVX 256bit = 8 float32
    
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                int w_out = 0;
                
                // 向量化处理
                for (; w_out <= W_out - vec_size; w_out += vec_size) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int k_h = 0; k_h < K_h; k_h++) {
                            for (int k_w = 0; k_w < K_w; k_w++) {
                                // 加载权重
                                int weight_idx = ((c_out * C_in + c_in) * K_h + k_h) * K_w + k_w;
                                __m256 weight_vec = _mm256_broadcast_ss(&weight[weight_idx]);
                                
                                // 加载输入向量
                                __m256 input_vec = _mm256_setzero_ps();
                                float input_vals[vec_size];
                                
                                for (int v = 0; v < vec_size; v++) {
                                    int h_in = h_out * stride_h - pad_h + k_h * dilation_h;
                                    int w_in = (w_out + v) * stride_w - pad_w + k_w * dilation_w;
                                    
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                                        input_vals[v] = input[input_idx];
                                    } else {
                                        input_vals[v] = 0.0f;
                                    }
                                }
                                
                                input_vec = _mm256_loadu_ps(input_vals);
                                
                                // FMA操作
                                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }
                    
                    // 添加偏置
                    {% if parameters.use_bias %}
                    if (bias != nullptr) {
                        __m256 bias_vec = _mm256_broadcast_ss(&bias[c_out]);
                        sum_vec = _mm256_add_ps(sum_vec, bias_vec);
                    }
                    {% endif %}
                    
                    // 存储结果
                    float results[vec_size];
                    _mm256_storeu_ps(results, sum_vec);
                    
                    for (int v = 0; v < vec_size && (w_out + v) < W_out; v++) {
                        int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + (w_out + v);
                        output[output_idx] = results[v];
                    }
                }
                
                // 处理剩余元素
                for (; w_out < W_out; w_out++) {
                    dtype_t sum = 0.0f;
                    
                    for (int c_in = 0; c_in < C_in; c_in++) {
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
                    
                    {% if parameters.use_bias %}
                    if (bias != nullptr) {
                        sum += bias[c_out];
                    }
                    {% endif %}
                    
                    int output_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output[output_idx] = sum;
                }
            }
        }
    }
}
{% endif %}