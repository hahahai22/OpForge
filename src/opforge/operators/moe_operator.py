"""
MoE（Mixture of Experts）算子生成器

支持生成高性能的稀疏MoE和密集MoE算子，包括专家选择、路由、
以及专家网络的高效实现。
"""

from typing import List, Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import math

from ..core.operator_base import OperatorBase, OperatorConfig, TensorShape, DataType, Backend


@dataclass
class MoEConfig(OperatorConfig):
    """MoE配置"""
    # 基本参数
    num_experts: int = 8          # 专家数量
    expert_dim: int = 512         # 专家网络维度
    hidden_dim: int = 2048        # 专家网络隐藏层维度
    top_k: int = 2               # TopK路由，选择前k个专家
    
    # 路由配置
    gate_type: str = "top_k"      # "top_k", "switch", "hash", "dense"
    gating_dim: int = None        # 门控网络输入维度（默认为expert_dim）
    load_balance_loss_weight: float = 0.01  # 负载均衡损失权重
    
    # 专家网络配置
    expert_type: str = "ffn"      # "ffn", "attention", "conv", "custom"
    activation: str = "relu"      # "relu", "gelu", "swish", "silu"
    dropout_rate: float = 0.1
    use_bias: bool = True
    
    # 性能优化
    use_expert_parallelism: bool = True    # 专家并行
    use_capacity_factor: bool = True       # 容量因子限制
    capacity_factor: float = 1.25         # 容量因子
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.gating_dim is None:
            self.gating_dim = self.expert_dim
            
        if self.num_experts <= 0:
            raise ValueError("专家数量必须大于0")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(f"top_k必须在1到{self.num_experts}之间")


class MoEOperator(OperatorBase):
    """MoE算子生成器"""
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.moe_config = config
        
        # 设置参数
        self.add_parameter("num_experts", config.num_experts)
        self.add_parameter("expert_dim", config.expert_dim)
        self.add_parameter("hidden_dim", config.hidden_dim)
        self.add_parameter("top_k", config.top_k)
        self.add_parameter("gating_dim", config.gating_dim)
    
    def get_operator_type(self) -> str:
        """返回算子类型"""
        if self.moe_config.gate_type == "top_k":
            return "sparse_moe"
        elif self.moe_config.gate_type == "dense":
            return "dense_moe"
        else:
            return "moe"
    
    def validate_config(self) -> bool:
        """验证配置"""
        config = self.moe_config
        
        if config.num_experts <= 0:
            return False
        if config.expert_dim <= 0 or config.hidden_dim <= 0:
            return False
        if config.top_k <= 0 or config.top_k > config.num_experts:
            return False
            
        return True
    
    def infer_output_shape(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """推断输出形状"""
        if not input_shapes:
            raise ValueError("需要至少一个输入张量")
        
        input_shape = input_shapes[0]
        if len(input_shape.dims) < 2:
            raise ValueError("输入张量至少需要2D (batch_size, expert_dim)")
        
        # MoE输出形状通常与输入相同
        output_shape = TensorShape(
            dims=input_shape.dims.copy(),
            dtype=input_shape.dtype,
            name="moe_output"
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
        return "// MoE反向传播代码较为复杂，建议使用框架提供的自动微分功能"
    
    def _generate_cuda_forward(self) -> str:
        """生成CUDA前向代码"""
        config = self.moe_config
        
        return f"""
// CUDA MoE前向传播主函数
__global__ void {self.get_kernel_name()}_forward(
    const float* input,           // [batch_size, seq_len, expert_dim]
    const float* gate_weights,    // [expert_dim, num_experts]
    const float* expert_w1,       // [num_experts, expert_dim, hidden_dim]
    const float* expert_w2,       // [num_experts, hidden_dim, expert_dim]
    float* output,               // [batch_size, seq_len, expert_dim]
    int batch_size,
    int seq_len,
    int expert_dim,
    int hidden_dim
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * seq_len;
    
    if (idx >= total_tokens) return;
    
    const float* input_token = input + idx * expert_dim;
    float* output_token = output + idx * expert_dim;
    
    // 1. 计算门控权重
    float gate_probs[{config.num_experts}];
    float max_logit = -INFINITY;
    
    // 计算logits
    for (int e = 0; e < {config.num_experts}; e++) {{
        float logit = 0.0f;
        for (int d = 0; d < expert_dim; d++) {{
            logit += input_token[d] * gate_weights[d * {config.num_experts} + e];
        }}
        gate_probs[e] = logit;
        max_logit = fmaxf(max_logit, logit);
    }}
    
    // Softmax
    float sum_exp = 0.0f;
    for (int e = 0; e < {config.num_experts}; e++) {{
        gate_probs[e] = expf(gate_probs[e] - max_logit);
        sum_exp += gate_probs[e];
    }}
    for (int e = 0; e < {config.num_experts}; e++) {{
        gate_probs[e] /= sum_exp;
    }}
    
    // 2. TopK选择
    int selected_experts[{config.top_k}];
    float selected_weights[{config.top_k}];
    
    for (int k = 0; k < {config.top_k}; k++) {{
        float max_prob = -1.0f;
        int max_idx = -1;
        
        for (int e = 0; e < {config.num_experts}; e++) {{
            bool already_selected = false;
            for (int prev_k = 0; prev_k < k; prev_k++) {{
                if (selected_experts[prev_k] == e) {{
                    already_selected = true;
                    break;
                }}
            }}
            
            if (!already_selected && gate_probs[e] > max_prob) {{
                max_prob = gate_probs[e];
                max_idx = e;
            }}
        }}
        
        selected_experts[k] = max_idx;
        selected_weights[k] = max_prob;
    }}
    
    // 重新归一化
    float weight_sum = 0.0f;
    for (int k = 0; k < {config.top_k}; k++) {{
        weight_sum += selected_weights[k];
    }}
    for (int k = 0; k < {config.top_k}; k++) {{
        selected_weights[k] /= weight_sum;
    }}
    
    // 3. 专家计算和聚合
    for (int d = 0; d < expert_dim; d++) {{
        output_token[d] = 0.0f;
    }}
    
    for (int k = 0; k < {config.top_k}; k++) {{
        int expert_id = selected_experts[k];
        float weight = selected_weights[k];
        
        if (expert_id < 0) continue;
        
        // FFN专家计算
        const float* w1 = expert_w1 + expert_id * expert_dim * hidden_dim;
        const float* w2 = expert_w2 + expert_id * hidden_dim * expert_dim;
        
        // 第一层
        float hidden[{min(config.hidden_dim, 512)}];  // 限制大小避免栈溢出
        for (int h = 0; h < hidden_dim && h < {min(config.hidden_dim, 512)}; h++) {{
            float sum = 0.0f;
            for (int d = 0; d < expert_dim; d++) {{
                sum += input_token[d] * w1[d * hidden_dim + h];
            }}
            hidden[h] = fmaxf(0.0f, sum);  // ReLU激活
        }}
        
        // 第二层
        for (int d = 0; d < expert_dim; d++) {{
            float sum = 0.0f;
            for (int h = 0; h < hidden_dim && h < {min(config.hidden_dim, 512)}; h++) {{
                sum += hidden[h] * w2[h * expert_dim + d];
            }}
            output_token[d] += sum * weight;
        }}
    }}
}}
"""
    
    def _generate_cpu_forward(self) -> str:
        """生成CPU前向代码"""
        config = self.moe_config
        
        return f"""
// CPU MoE前向传播实现
void {self.get_kernel_name()}_forward_cpu(
    const float* input,
    const float* gate_weights,
    const float* expert_weights,
    float* output,
    int batch_size,
    int seq_len,
    int expert_dim,
    int hidden_dim
) {{
    #pragma omp parallel for collapse(2)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
        for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {{
            int token_idx = batch_idx * seq_len + seq_idx;
            const float* input_token = input + token_idx * expert_dim;
            float* output_token = output + token_idx * expert_dim;
            
            // 简化的MoE计算
            // 1. 计算门控权重
            float gate_probs[{config.num_experts}];
            // ... 省略详细实现
            
            // 2. 专家选择和计算
            // ... 省略详细实现
            
            // 3. 结果聚合
            for (int d = 0; d < expert_dim; d++) {{
                output_token[d] = input_token[d];  // 占位实现
            }}
        }}
    }}
}}
"""
    
    def get_performance_hints(self) -> List[str]:
        """获取性能优化建议"""
        hints = super().get_performance_hints()
        
        config = self.moe_config
        
        # MoE特定的性能建议
        if config.top_k >= config.num_experts / 2:
            hints.append("TopK值较大，考虑使用密集MoE")
        
        if config.num_experts > 64:
            hints.append("专家数量较多，建议使用专家并行策略")
        
        if config.hidden_dim > 4096:
            hints.append("隐藏层维度较大，建议使用更多的内存优化")
        
        return hints