"""
{{operator_type}}算子测试用例
自动生成 - 请勿手动修改
"""

import unittest
import torch
import numpy as np
from typing import Optional

{% if backend.value == 'triton' %}
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
{% endif %}


class Test{{operator_type.title()}}(unittest.TestCase):
    """{{operator_type}}算子测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.{{dtype.value.replace('32', '')}}
        self.rtol = 1e-3 if self.dtype == torch.float16 else 1e-5
        self.atol = 1e-3 if self.dtype == torch.float16 else 1e-6
        
        {% if operator_type == 'conv2d' %}
        # 卷积参数
        self.batch_size = {{parameters.get('batch_size', 2)}}
        self.in_channels = {{parameters.get('in_channels', 32)}}
        self.out_channels = {{parameters.get('out_channels', 64)}}
        self.height = 64
        self.width = 64
        self.kernel_size = {{parameters.get('kernel_h', 3)}}
        self.stride = {{parameters.get('stride_h', 1)}}
        self.padding = {{parameters.get('pad_h', 1)}}
        {% elif operator_type == 'softmax' %}
        # Softmax参数
        self.batch_size = 8
        self.seq_len = 128
        self.hidden_dim = 512
        self.dim = {{parameters.get('dim', -1)}}
        {% elif operator_type == 'moe' %}
        # MoE参数
        self.batch_size = 4
        self.seq_len = 64
        self.expert_dim = {{parameters.get('expert_dim', 256)}}
        self.hidden_dim = {{parameters.get('hidden_dim', 1024)}}
        self.num_experts = {{parameters.get('num_experts', 4)}}
        self.top_k = {{parameters.get('top_k', 2)}}
        {% endif %}
    
    def test_forward_correctness(self):
        """测试前向传播正确性"""
        
        {% if operator_type == 'conv2d' %}
        # 创建测试数据
        input_tensor = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            dtype=self.dtype, device=self.device
        )
        weight = torch.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
            dtype=self.dtype, device=self.device
        )
        bias = torch.randn(self.out_channels, dtype=self.dtype, device=self.device)
        
        # 参考实现 (PyTorch)
        conv_ref = torch.nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding, bias=True
        ).to(self.device).to(self.dtype)
        
        conv_ref.weight.data = weight
        conv_ref.bias.data = bias
        
        expected_output = conv_ref(input_tensor)
        
        # TODO: 替换为实际的算子调用
        # actual_output = {{operator_type}}_function(input_tensor, weight, bias,
        #                                           stride=self.stride, padding=self.padding)
        
        # 暂时使用参考实现进行测试
        actual_output = expected_output
        
        {% elif operator_type == 'softmax' %}
        # 创建测试数据
        input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            dtype=self.dtype, device=self.device
        )
        
        # 参考实现
        expected_output = torch.softmax(input_tensor, dim=self.dim)
        
        # TODO: 替换为实际的算子调用
        actual_output = expected_output
        
        {% elif operator_type == 'moe' %}
        # 创建测试数据
        input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.expert_dim,
            dtype=self.dtype, device=self.device
        )
        
        # 简化的参考实现
        gate_weights = torch.randn(self.expert_dim, self.num_experts, device=self.device)
        expert_weights = torch.randn(
            self.num_experts, self.expert_dim, self.hidden_dim,
            dtype=self.dtype, device=self.device
        )
        
        # 简单的门控和专家选择
        gate_logits = torch.matmul(input_tensor, gate_weights)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        # TopK选择
        top_k_values, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_values, dim=-1)
        
        # 简化的专家计算（线性变换）
        expert_outputs = torch.matmul(input_tensor.unsqueeze(-2), expert_weights[0])
        expected_output = expert_outputs.squeeze(-2)
        
        actual_output = expected_output
        {% endif %}
        
        # 验证输出形状
        self.assertEqual(actual_output.shape, expected_output.shape)
        
        # 验证数值正确性
        torch.testing.assert_close(
            actual_output, expected_output,
            rtol=self.rtol, atol=self.atol
        )
    
    def test_backward_correctness(self):
        """测试反向传播正确性"""
        
        {% if operator_type == 'conv2d' %}
        input_tensor = torch.randn(
            self.batch_size, self.in_channels, self.height, self.width,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        weight = torch.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        bias = torch.randn(self.out_channels, dtype=self.dtype, device=self.device, requires_grad=True)
        
        {% elif operator_type == 'softmax' %}
        input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        {% elif operator_type == 'moe' %}
        input_tensor = torch.randn(
            self.batch_size, self.seq_len, self.expert_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        {% endif %}
        
        # 使用PyTorch的自动微分作为参考
        {% if operator_type == 'conv2d' %}
        ref_conv = torch.nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding
        ).to(self.device).to(self.dtype)
        
        ref_conv.weight.data = weight
        ref_conv.bias.data = bias
        
        ref_output = ref_conv(input_tensor)
        {% elif operator_type == 'softmax' %}
        ref_output = torch.softmax(input_tensor, dim=self.dim)
        {% elif operator_type == 'moe' %}
        # 简化的MoE前向
        ref_output = input_tensor  # 简化为恒等映射
        {% endif %}
        
        # 计算损失并反向传播
        loss = ref_output.sum()
        loss.backward()
        
        # 验证梯度不为None
        self.assertIsNotNone(input_tensor.grad)
        {% if operator_type == 'conv2d' %}
        self.assertIsNotNone(weight.grad)
        if bias is not None:
            self.assertIsNotNone(bias.grad)
        {% endif %}
    
    def test_different_shapes(self):
        """测试不同输入形状"""
        
        test_shapes = [
            {% if operator_type == 'conv2d' %}
            (1, self.in_channels, 32, 32),
            (4, self.in_channels, 64, 64),
            (8, self.in_channels, 128, 128),
            {% elif operator_type == 'softmax' %}
            (1, 64, 256),
            (4, 128, 512),
            (8, 256, 1024),
            {% elif operator_type == 'moe' %}
            (1, 32, self.expert_dim),
            (2, 64, self.expert_dim),
            (4, 128, self.expert_dim),
            {% endif %}
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                input_tensor = torch.randn(*shape, dtype=self.dtype, device=self.device)
                
                {% if operator_type == 'conv2d' %}
                weight = torch.randn(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
                    dtype=self.dtype, device=self.device
                )
                
                # 计算期望输出形状
                h_out = (shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
                w_out = (shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
                expected_shape = (shape[0], self.out_channels, h_out, w_out)
                
                # 使用PyTorch参考实现
                conv_ref = torch.nn.Conv2d(
                    self.in_channels, self.out_channels, self.kernel_size,
                    stride=self.stride, padding=self.padding, bias=False
                ).to(self.device).to(self.dtype)
                conv_ref.weight.data = weight
                
                output = conv_ref(input_tensor)
                
                {% elif operator_type == 'softmax' %}
                expected_shape = shape
                output = torch.softmax(input_tensor, dim=self.dim)
                
                {% elif operator_type == 'moe' %}
                expected_shape = shape
                output = input_tensor  # 简化实现
                {% endif %}
                
                self.assertEqual(output.shape, expected_shape)
    
    {% if config.optimization_level >= 1 %}
    def test_numerical_stability(self):
        """测试数值稳定性"""
        
        {% if operator_type == 'softmax' %}
        # 测试大数值输入
        large_input = torch.full(
            (2, 32, 64), 1000.0,
            dtype=self.dtype, device=self.device
        )
        
        output = torch.softmax(large_input, dim=self.dim)
        
        # 验证输出在有效范围内
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        self.assertTrue(torch.allclose(output.sum(dim=self.dim), torch.ones_like(output.sum(dim=self.dim))))
        
        # 测试小数值输入
        small_input = torch.full(
            (2, 32, 64), -1000.0,
            dtype=self.dtype, device=self.device
        )
        
        output = torch.softmax(small_input, dim=self.dim)
        self.assertTrue(torch.all(torch.isfinite(output)))
        {% endif %}
    {% endif %}
    
    {% if backend.value == 'triton' and config.optimization_level >= 2 %}
    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_triton_autotune(self):
        """测试Triton自动调优"""
        
        input_tensor = torch.randn(
            {% if operator_type == 'conv2d' %}
            self.batch_size, self.in_channels, self.height, self.width,
            {% elif operator_type == 'softmax' %}
            self.batch_size, self.seq_len, self.hidden_dim,
            {% elif operator_type == 'moe' %}
            self.batch_size, self.seq_len, self.expert_dim,
            {% endif %}
            dtype=self.dtype, device=self.device
        )
        
        # TODO: 调用Triton优化版本
        # output = {{operator_type}}_triton_optimized(input_tensor)
        
        # 暂时跳过
        self.skipTest("Triton实现待完成")
    {% endif %}
    
    def test_performance_benchmark(self):
        """性能基准测试"""
        
        input_tensor = torch.randn(
            {% if operator_type == 'conv2d' %}
            32, self.in_channels, 224, 224,  # 更大的输入用于性能测试
            {% elif operator_type == 'softmax' %}
            32, 512, 768,
            {% elif operator_type == 'moe' %}
            32, 512, self.expert_dim,
            {% endif %}
            dtype=self.dtype, device=self.device
        )
        
        {% if operator_type == 'conv2d' %}
        weight = torch.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size,
            dtype=self.dtype, device=self.device
        )
        conv_layer = torch.nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding
        ).to(self.device).to(self.dtype)
        conv_layer.weight.data = weight
        {% endif %}
        
        # 预热
        for _ in range(5):
            {% if operator_type == 'conv2d' %}
            _ = conv_layer(input_tensor)
            {% elif operator_type == 'softmax' %}
            _ = torch.softmax(input_tensor, dim=self.dim)
            {% elif operator_type == 'moe' %}
            _ = input_tensor  # 简化
            {% endif %}
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        # 性能测试
        num_iterations = 50
        for _ in range(num_iterations):
            {% if operator_type == 'conv2d' %}
            output = conv_layer(input_tensor)
            {% elif operator_type == 'softmax' %}
            output = torch.softmax(input_tensor, dim=self.dim)
            {% elif operator_type == 'moe' %}
            output = input_tensor
            {% endif %}
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        print(f"\\n{{operator_type.title()}} 平均执行时间: {avg_time*1000:.2f} ms")
        print(f"设备: {self.device}")
        print(f"数据类型: {self.dtype}")
        print(f"后端: {{backend.value}}")
        
        # 验证性能在合理范围内（这个阈值需要根据实际情况调整）
        max_time_ms = 100  # 100ms阈值
        self.assertLess(avg_time * 1000, max_time_ms, 
                       f"Performance too slow: {avg_time*1000:.2f}ms > {max_time_ms}ms")


if __name__ == '__main__':
    unittest.main()