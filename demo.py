#!/usr/bin/env python3
"""
OpForge 完整演示

展示所有主要功能的演示脚本。
"""

import sys
from pathlib import Path

print("🚀 OpForge - 深度学习算子自动生成工具")
print("=" * 60)
print()

# 检查依赖
try:
    from opforge.core import Backend, DataType, TensorShape, BackendManager
    from opforge.operators import Conv2DOperator, SoftmaxOperator, MoEOperator
    from opforge.operators.conv_operator import Conv2DConfig
    from opforge.operators.softmax_operator import SoftmaxConfig
    from opforge.operators.moe_operator import MoEConfig
    from opforge.core.code_generator import CodeGenerator
    
    print("✅ 所有依赖模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

print()

# 1. 检查可用后端
print("📋 第一步：检查可用的计算后端")
print("-" * 40)
manager = BackendManager()
backends = manager.get_available_backends()

for backend in backends:
    capability = manager.get_backend_capability(backend)
    print(f"  • {backend.value}: {capability.name} v{capability.version}")
    
    features = []
    if capability.supports_fp16:
        features.append("FP16")
    if capability.supports_int8:
        features.append("INT8")
    if capability.compute_capability:
        features.append(f"Compute {capability.compute_capability}")
        
    if features:
        print(f"    特性: {', '.join(features)}")

optimal_backend = manager.get_optimal_backend()
print(f"\n🎯 推荐后端: {optimal_backend.value}")
print()

# 2. 测试Conv2D算子
print("📋 第二步：生成2D卷积算子")
print("-" * 40)

conv_config = Conv2DConfig(
    name="demo_conv2d",
    backend=optimal_backend,
    dtype=DataType.FLOAT32,
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    optimization_level=2
)

conv_operator = Conv2DOperator(conv_config)
print(f"✅ 创建算子: {conv_operator.get_operator_type()}")

# 设置输入形状
input_shape = TensorShape(
    dims=[8, 64, 224, 224],  # [batch, channels, height, width]
    dtype=DataType.FLOAT32,
    name="input"
)

conv_operator.set_input_shapes([input_shape])
print(f"📐 输入形状: {input_shape}")
print(f"📐 输出形状: {conv_operator.output_shapes[0]}")

# 内存分析
memory_req = conv_operator.get_memory_requirements()
print(f"💾 内存需求: {memory_req['total_memory_bytes'] / 1024 / 1024:.2f} MB")

# 性能建议
hints = conv_operator.get_performance_hints()
if hints:
    print("💡 性能建议:")
    for hint in hints:
        print(f"  • {hint}")

print()

# 3. 测试Softmax算子
print("📋 第三步：生成Softmax算子")
print("-" * 40)

softmax_config = SoftmaxConfig(
    name="demo_softmax",
    backend=optimal_backend,
    dtype=DataType.FLOAT32,
    dim=-1,
    temperature=1.0,
    use_log_softmax=False,
    optimization_level=2
)

softmax_operator = SoftmaxOperator(softmax_config)
print(f"✅ 创建算子: {softmax_operator.get_operator_type()}")

# 设置输入形状
softmax_input_shape = TensorShape(
    dims=[32, 128, 512],  # [batch, seq_len, hidden_dim]
    dtype=DataType.FLOAT32,
    name="attention_scores"
)

softmax_operator.set_input_shapes([softmax_input_shape])
print(f"📐 输入形状: {softmax_input_shape}")
print(f"📐 输出形状: {softmax_operator.output_shapes[0]}")

print()

# 4. 测试MoE算子
print("📋 第四步：生成MoE算子")
print("-" * 40)

moe_config = MoEConfig(
    name="demo_moe",
    backend=optimal_backend,
    dtype=DataType.FLOAT32,
    num_experts=8,
    expert_dim=512,
    hidden_dim=2048,
    top_k=2,
    gate_type="top_k",
    expert_type="ffn",
    optimization_level=2
)

moe_operator = MoEOperator(moe_config)
print(f"✅ 创建算子: {moe_operator.get_operator_type()}")

# 设置输入形状
moe_input_shape = TensorShape(
    dims=[16, 64, 512],  # [batch, seq_len, expert_dim]
    dtype=DataType.FLOAT32,
    name="hidden_states"
)

moe_operator.set_input_shapes([moe_input_shape])
print(f"📐 输入形状: {moe_input_shape}")
print(f"📐 输出形状数量: {len(moe_operator.output_shapes)}")

# 计算稀疏比率
sparsity_ratio = moe_config.top_k / moe_config.num_experts
print(f"⚡ 稀疏比率: {sparsity_ratio:.2%} (选择 {moe_config.top_k}/{moe_config.num_experts} 专家)")

print()

# 5. 代码生成演示
print("📋 第五步：代码生成演示")
print("-" * 40)

generator = CodeGenerator()

# 尝试生成代码（可能没有模板文件）
operators = [
    ("Conv2D", conv_operator),
    ("Softmax", softmax_operator), 
    ("MoE", moe_operator)
]

for name, operator in operators:
    try:
        # 生成前向代码（直接调用算子方法）
        forward_code = operator.generate_forward_code()
        if forward_code and len(forward_code) > 50:
            print(f"✅ {name} 前向代码生成成功 ({len(forward_code)} 字符)")
        else:
            print(f"⚠️  {name} 前向代码生成为空或太短")
            
        # 生成后向代码
        backward_code = operator.generate_backward_code()
        if backward_code and len(backward_code) > 10:
            print(f"✅ {name} 反向代码生成成功 ({len(backward_code)} 字符)")
        else:
            print(f"⚠️  {name} 反向代码生成为空或太短")
            
    except Exception as e:
        print(f"❌ {name} 代码生成失败: {e}")

print()

# 6. 总结
print("📋 第六步：总结")
print("-" * 40)

print("🎉 OpForge 演示完成！")
print()
print("✨ 主要功能:")
print("  • ✅ 多后端支持 (CPU, CUDA, OpenCL)")
print("  • ✅ 多种算子类型 (Conv2D, Softmax, MoE)")
print("  • ✅ 自动形状推断")
print("  • ✅ 内存需求分析")
print("  • ✅ 性能优化建议")
print("  • ✅ 代码自动生成")
print("  • ✅ 模板化架构")
print("  • ✅ 命令行工具")
print()
print("🚀 现在您可以使用 OpForge 生成高性能的深度学习算子!")
print()
print("📖 更多信息:")
print("  • 查看 docs/USER_GUIDE.md 了解详细使用方法")
print("  • 运行 examples/ 目录下的示例")
print("  • 使用 'opforge --help' 查看命令行选项")
print("  • 查看 tests/ 目录了解如何运行测试")

if __name__ == "__main__":
    pass