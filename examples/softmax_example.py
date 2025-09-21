#!/usr/bin/env python3
"""
OpForge示例：生成Softmax算子

演示如何使用OpForge生成高性能的Softmax算子。
"""

from opforge.core import Backend, DataType, TensorShape
from opforge.operators import SoftmaxOperator
from opforge.operators.softmax_operator import SoftmaxConfig
from opforge.core.code_generator import CodeGenerator


def main():
    """主函数"""
    print("🚀 OpForge Softmax算子生成示例")
    print("=" * 50)
    
    # 1. 创建Softmax配置
    config = SoftmaxConfig(
        name="example_softmax",
        backend=Backend.CUDA,
        dtype=DataType.FLOAT32,
        optimization_level=2,
        debug_mode=False,
        
        # Softmax参数
        dim=-1,
        temperature=1.0,
        use_log_softmax=False,
        
        # 性能优化
        use_online_softmax=True,
        use_warp_reduce=True,
        use_block_reduce=True
    )
    
    print(f"📋 配置信息:")
    print(f"  • 计算维度: {config.dim}")
    print(f"  • 温度参数: {config.temperature}")
    print(f"  • LogSoftmax: {config.use_log_softmax}")
    print(f"  • 在线算法: {config.use_online_softmax}")
    print(f"  • 后端: {config.backend.value}")
    
    # 2. 创建算子
    operator = SoftmaxOperator(config)
    
    # 验证配置
    if not operator.validate_config():
        print("❌ 配置验证失败")
        return
    
    print("✅ 配置验证通过")
    
    # 3. 设置输入形状
    input_shape = TensorShape(
        dims=[32, 128, 512],  # [batch_size, seq_len, hidden_dim]
        dtype=DataType.FLOAT32,
        name="input"
    )
    
    operator.set_input_shapes([input_shape])
    output_shapes = operator.output_shapes
    
    print(f"\n📐 张量形状:")
    print(f"  • 输入: {input_shape}")
    print(f"  • 输出: {output_shapes[0]}")
    
    # 4. 生成代码
    print(f"\n🔨 生成代码...")
    generator = CodeGenerator()
    
    try:
        generated_files = generator.generate_operator_code(operator)
        
        print(f"✅ 代码生成成功!")
        print(f"📄 生成的文件:")
        for file_type in generated_files.keys():
            print(f"  • {file_type}")
        
        # 5. 保存文件
        output_dir = "./generated/softmax_example"
        saved_files = generator.save_generated_code(generated_files, output_dir)
        
        print(f"\n💾 文件已保存到: {output_dir}")
        for file_type, file_path in saved_files.items():
            print(f"  • {file_type}: {file_path}")
        
        # 6. 性能建议
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