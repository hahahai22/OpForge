#!/usr/bin/env python3
"""
OpForge示例：生成MoE算子

演示如何使用OpForge生成高性能的MoE算子。
"""

from opforge.core import Backend, DataType, TensorShape
from opforge.operators import MoEOperator
from opforge.operators.moe_operator import MoEConfig
from opforge.core.code_generator import CodeGenerator


def main():
    """主函数"""
    print("🚀 OpForge MoE算子生成示例")
    print("=" * 50)
    
    # 1. 创建MoE配置
    config = MoEConfig(
        name="example_moe",
        backend=Backend.CUDA,
        dtype=DataType.FLOAT32,
        optimization_level=2,
        debug_mode=False,
        
        # MoE参数
        num_experts=8,
        expert_dim=512,
        hidden_dim=2048,
        top_k=2,
        
        # 路由配置
        gate_type="top_k",
        gating_dim=512,
        load_balance_loss_weight=0.01,
        
        # 专家配置
        expert_type="ffn",
        activation="relu",
        use_bias=True,
        
        # 性能优化
        use_expert_parallelism=True,
        use_capacity_factor=True,
        capacity_factor=1.25
    )
    
    print(f"📋 配置信息:")
    print(f"  • 专家数量: {config.num_experts}")
    print(f"  • 专家维度: {config.expert_dim}")
    print(f"  • 隐藏层维度: {config.hidden_dim}")
    print(f"  • TopK: {config.top_k}")
    print(f"  • 门控类型: {config.gate_type}")
    print(f"  • 专家类型: {config.expert_type}")
    print(f"  • 后端: {config.backend.value}")
    
    # 2. 创建算子
    operator = MoEOperator(config)
    
    # 验证配置
    if not operator.validate_config():
        print("❌ 配置验证失败")
        return
    
    print("✅ 配置验证通过")
    
    # 3. 设置输入形状
    input_shape = TensorShape(
        dims=[16, 128, 512],  # [batch_size, seq_len, expert_dim]
        dtype=DataType.FLOAT32,
        name="input"
    )
    
    operator.set_input_shapes([input_shape])
    output_shapes = operator.output_shapes
    
    print(f"\n📐 张量形状:")
    print(f"  • 输入: {input_shape}")
    for i, output_shape in enumerate(output_shapes):
        print(f"  • 输出{i+1}: {output_shape}")
    
    # 4. 计算复杂度
    batch_size, seq_len, expert_dim = 16, 128, 512
    total_tokens = batch_size * seq_len
    expert_flops = config.expert_dim * config.hidden_dim * 2  # FFN FLOPs
    total_flops = total_tokens * config.top_k * expert_flops
    
    print(f"\n🔢 计算复杂度:")
    print(f"  • 总token数: {total_tokens}")
    print(f"  • 每个专家FLOPs: {expert_flops:,}")
    print(f"  • 总FLOPs: {total_flops:,}")
    print(f"  • 稀疏比率: {config.top_k/config.num_experts:.2%}")
    
    # 5. 生成代码
    print(f"\n🔨 生成代码...")
    generator = CodeGenerator()
    
    try:
        generated_files = generator.generate_operator_code(operator)
        
        print(f"✅ 代码生成成功!")
        print(f"📄 生成的文件:")
        for file_type in generated_files.keys():
            print(f"  • {file_type}")
        
        # 6. 保存文件
        output_dir = "./generated/moe_example"
        saved_files = generator.save_generated_code(generated_files, output_dir)
        
        print(f"\n💾 文件已保存到: {output_dir}")
        for file_type, file_path in saved_files.items():
            print(f"  • {file_type}: {file_path}")
        
        # 7. 性能建议
        hints = operator.get_performance_hints()
        if hints:
            print(f"\n💡 性能优化建议:")
            for hint in hints:
                print(f"  • {hint}")
        
        print(f"\n🎉 示例完成!")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")


if __name__ == "__main__":
    main()