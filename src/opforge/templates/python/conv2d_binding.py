"""
Python绑定模板 - {{operator_type}}算子
自动生成 - 请勿手动修改
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

{% if backend.value == 'triton' %}
import triton
import triton.language as tl
{% endif %}

class {{operator_type.title()}}Op(torch.autograd.Function):
    """{{operator_type}}算子的PyTorch Function实现"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, 
                bias: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """前向传播"""
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(input, weight, bias)
        ctx.kwargs = kwargs
        
        {% if backend.value == 'cuda' %}
        # 调用CUDA内核
        return {{kernel_name}}_cuda_forward(input, weight, bias, **kwargs)
        {% elif backend.value == 'triton' %}
        # 调用Triton内核
        return {{kernel_name}}_triton_launcher(input, weight, bias, **kwargs)
        {% elif backend.value == 'cpu' %}
        # 调用CPU实现
        return {{kernel_name}}_cpu_forward(input, weight, bias, **kwargs)
        {% else %}
        raise NotImplementedError(f"Backend {backend.value} not implemented")
        {% endif %}
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """反向传播"""
        input, weight, bias = ctx.saved_tensors
        kwargs = ctx.kwargs
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # 计算输入梯度
            grad_input = {{kernel_name}}_backward_input(weight, grad_output, **kwargs)
        
        if ctx.needs_input_grad[1]:
            # 计算权重梯度
            grad_weight = {{kernel_name}}_backward_weight(input, grad_output, **kwargs)
        
        if ctx.needs_input_grad[2] and bias is not None:
            # 计算偏置梯度
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        return grad_input, grad_weight, grad_bias


class {{operator_type.title()}}Layer(nn.Module):
    """{{operator_type}}算子的PyTorch Layer封装"""
    
    def __init__(self, 
                 {% if operator_type == 'conv2d' %}
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True
                 {% elif operator_type == 'softmax' %}
                 dim: int = -1,
                 temperature: float = 1.0
                 {% elif operator_type == 'moe' %}
                 num_experts: int = 8,
                 expert_dim: int = 512,
                 hidden_dim: int = 2048,
                 top_k: int = 2
                 {% endif %}
                 ):
        super().__init__()
        
        {% if operator_type == 'conv2d' %}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # 初始化参数
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
        {% elif operator_type == 'softmax' %}
        self.dim = dim
        self.temperature = temperature
        
        {% elif operator_type == 'moe' %}
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        # 门控网络
        self.gate = nn.Linear(expert_dim, num_experts)
        
        # 专家网络权重
        self.expert_w1 = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim))
        self.expert_w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim))
        self.expert_b1 = nn.Parameter(torch.randn(num_experts, hidden_dim))
        self.expert_b2 = nn.Parameter(torch.randn(num_experts, expert_dim))
        
        # 初始化参数
        nn.init.normal_(self.expert_w1, std=0.02)
        nn.init.normal_(self.expert_w2, std=0.02)
        nn.init.zeros_(self.expert_b1)
        nn.init.zeros_(self.expert_b2)
        {% endif %}
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        
        {% if operator_type == 'conv2d' %}
        return {{operator_type.title()}}Op.apply(
            input, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )
        {% elif operator_type == 'softmax' %}
        return {{operator_type.title()}}Op.apply(
            input, dim=self.dim, temperature=self.temperature
        )
        {% elif operator_type == 'moe' %}
        return {{operator_type.title()}}Op.apply(
            input, 
            self.expert_w1, self.expert_w2,
            self.expert_b1, self.expert_b2,
            self.gate.weight, self.gate.bias,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        {% endif %}


# 便利函数
def {{operator_type}}_function(input: torch.Tensor, 
                              {% if operator_type == 'conv2d' %}
                              weight: torch.Tensor,
                              bias: Optional[torch.Tensor] = None,
                              stride: int = 1,
                              padding: int = 0
                              {% elif operator_type == 'softmax' %}
                              dim: int = -1,
                              temperature: float = 1.0
                              {% elif operator_type == 'moe' %}
                              expert_weights: torch.Tensor,
                              gate_weights: torch.Tensor,
                              num_experts: int = 8,
                              top_k: int = 2
                              {% endif %}
                              ) -> torch.Tensor:
    """{{operator_type}}算子的便利函数接口"""
    
    {% if operator_type == 'conv2d' %}
    return {{operator_type.title()}}Op.apply(input, weight, bias, stride=stride, padding=padding)
    {% elif operator_type == 'softmax' %}
    return {{operator_type.title()}}Op.apply(input, dim=dim, temperature=temperature)
    {% elif operator_type == 'moe' %}
    return {{operator_type.title()}}Op.apply(input, expert_weights, gate_weights, 
                                          num_experts=num_experts, top_k=top_k)
    {% endif %}


# 性能基准测试
def benchmark_{{operator_type}}(
    {% if operator_type == 'conv2d' %}
    batch_size: int = 32,
    in_channels: int = 64,
    out_channels: int = 128,
    height: int = 224,
    width: int = 224,
    kernel_size: int = 3
    {% elif operator_type == 'softmax' %}
    batch_size: int = 32,
    seq_len: int = 512,
    hidden_dim: int = 768
    {% elif operator_type == 'moe' %}
    batch_size: int = 32,
    seq_len: int = 512,
    expert_dim: int = 512,
    num_experts: int = 8
    {% endif %}
) -> dict:
    """性能基准测试"""
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    {% if operator_type == 'conv2d' %}
    # 创建测试数据
    input_tensor = torch.randn(batch_size, in_channels, height, width, device=device)
    layer = {{operator_type.title()}}Layer(in_channels, out_channels, kernel_size).to(device)
    {% elif operator_type == 'softmax' %}
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    layer = {{operator_type.title()}}Layer(dim=-1).to(device)
    {% elif operator_type == 'moe' %}
    input_tensor = torch.randn(batch_size, seq_len, expert_dim, device=device)
    layer = {{operator_type.title()}}Layer(num_experts, expert_dim, expert_dim*4, 2).to(device)
    {% endif %}
    
    # 预热
    for _ in range(10):
        _ = layer(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 性能测试
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        output = layer(input_tensor)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    
    return {
        'average_time_ms': avg_time * 1000,
        'throughput_samples_per_sec': batch_size / avg_time,
        'device': str(device),
        'backend': '{{backend.value}}',
        'optimization_level': {{optimization_level}},
        {% if config.enable_tensor_core %}
        'tensor_core_enabled': True,
        {% endif %}
        {% if config.enable_dot_instructions %}
        'dot_instructions_enabled': True,
        {% endif %}
    }


if __name__ == '__main__':
    # 运行基准测试
    results = benchmark_{{operator_type}}()
    print(f"{{operator_type.title()}} 性能测试结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")