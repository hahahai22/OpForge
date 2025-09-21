#!/usr/bin/env python3
"""
OpForge示例：生成2D卷积算子

演示如何使用OpForge生成高性能的2D卷积算子。
"""

from opforge.core import Backend, DataType, TensorShape
from opforge.operators import Conv2DOperator
from opforge.operators.conv_operator import Conv2DConfig
from opforge.core.code_generator import CodeGenerator


def main():
    """主函数"""
    print("🚀 OpForge 2D卷积算子生成示例")
    print("=" * 50)
    
    # 1. 创建卷积配置
    config = Conv2DConfig(
        name="example_conv2d",
        backend=Backend.CPU,  # 使用CPU后端确保可用
        dtype=DataType.FLOAT32,
        optimization_level=2,
        debug_mode=False,
        
        # 卷积参数
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=True,
        
        # 性能优化
        use_winograd=False,
        use_im2col=True
    )
    
    print(f"📋 配置信息:")
    print(f"  • 输入通道: {config.in_channels}")
    print(f"  • 输出通道: {config.out_channels}")
    print(f"  • 卷积核大小: {config.kernel_size}")
    print(f"  • 后端: {config.backend.value}")
    print(f"  • 数据类型: {config.dtype.value}")
    
    # 2. 创建算子
    operator = Conv2DOperator(config)
    
    # 验证配置
    if not operator.validate_config():
        print("❌ 配置验证失败")
        return
    
    print("✅ 配置验证通过")
    
    # 3. 设置输入形状
    input_shape = TensorShape(
        dims=[1, 64, 224, 224],  # [N, C, H, W]
        dtype=DataType.FLOAT32,
        name="input"
    )
    
    operator.set_input_shapes([input_shape])
    output_shapes = operator.output_shapes
    
    print(f"\n📐 张量形状:")
    print(f"  • 输入: {input_shape}")
    print(f"  • 输出: {output_shapes[0]}")
    
    # 4. 内存需求分析
    memory_req = operator.get_memory_requirements()
    print(f"\n💾 内存需求:")
    print(f"  • 输入内存: {memory_req['input_memory_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  • 输出内存: {memory_req['output_memory_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  • 总内存: {memory_req['total_memory_bytes'] / 1024 / 1024:.2f} MB")
    
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
        output_dir = "./generated/conv2d_example"
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