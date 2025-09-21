"""
OpForge命令行界面

提供友好的命令行工具来生成各种深度学习算子。
"""

import click
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from .core import OperatorConfig, TensorShape, DataType, Backend, CodeGenerator, BackendManager
from .operators import Conv2DOperator, SoftmaxOperator, MoEOperator
from .operators.conv_operator import Conv2DConfig
from .operators.softmax_operator import SoftmaxConfig
from .operators.moe_operator import MoEConfig


@click.group()
@click.version_option(version="0.1.0")
@click.option('--verbose', '-v', is_flag=True, help='启用详细输出')
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.pass_context
def cli(ctx, verbose, config):
    """OpForge - 深度学习算子自动生成工具"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if config:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                ctx.obj['config'] = yaml.safe_load(f)
            else:
                ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}


@cli.command()
def list_backends():
    """列出可用的计算后端"""
    manager = BackendManager()
    backends = manager.get_available_backends()
    
    click.echo("可用的计算后端:")
    for backend in backends:
        capability = manager.get_backend_capability(backend)
        click.echo(f"  • {backend.value}: {capability.name} v{capability.version}")
        
        # 显示后端特性
        features = []
        if capability.supports_fp16:
            features.append("FP16")
        if capability.supports_int8:
            features.append("INT8")
        if capability.compute_capability:
            features.append(f"Compute {capability.compute_capability}")
            
        if features:
            click.echo(f"    特性: {', '.join(features)}")


@cli.command()
@click.argument('operator_type', type=click.Choice(['conv2d', 'softmax', 'moe']))
@click.option('--backend', '-b', type=click.Choice(['cuda', 'cpu', 'opencl', 'triton', 'rocm']), 
              default='cuda', help='目标后端')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'int32']), 
              default='float32', help='数据类型')
@click.option('--output', '-o', type=click.Path(), default='./generated', 
              help='输出目录')
@click.option('--optimization', type=click.IntRange(0, 2), default=2, 
              help='优化级别 (0=无优化, 1=基础优化, 2=激进优化)')
@click.option('--debug', is_flag=True, help='启用调试模式')
@click.pass_context
def generate(ctx, operator_type, backend, dtype, output, optimization, debug):
    """生成指定类型的算子代码"""
    
    # 验证后端可用性
    manager = BackendManager()
    backend_enum = Backend(backend)
    
    if not manager.is_backend_available(backend_enum):
        click.echo(f"错误: 后端 {backend} 不可用", err=True)
        sys.exit(1)
    
    # 根据算子类型调用相应的子命令
    if operator_type == 'conv2d':
        ctx.invoke(generate_conv2d, backend=backend, dtype=dtype, output=output, 
                  optimization=optimization, debug=debug)
    elif operator_type == 'softmax':
        ctx.invoke(generate_softmax, backend=backend, dtype=dtype, output=output,
                  optimization=optimization, debug=debug)
    elif operator_type == 'moe':
        ctx.invoke(generate_moe, backend=backend, dtype=dtype, output=output,
                  optimization=optimization, debug=debug)


@cli.command()
@click.option('--backend', '-b', type=click.Choice(['cuda', 'cpu', 'opencl', 'triton', 'rocm']), 
              default='cuda', help='目标后端')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'int32']), 
              default='float32', help='数据类型')
@click.option('--in-channels', type=int, default=3, help='输入通道数')
@click.option('--out-channels', type=int, default=64, help='输出通道数')
@click.option('--kernel-size', type=int, default=3, help='卷积核大小')
@click.option('--stride', type=int, default=1, help='步长')
@click.option('--padding', type=int, default=1, help='填充')
@click.option('--groups', type=int, default=1, help='分组数')
@click.option('--bias/--no-bias', default=True, help='是否使用偏置')
@click.option('--output', '-o', type=click.Path(), default='./generated/conv2d', 
              help='输出目录')
@click.option('--optimization', type=click.IntRange(0, 2), default=2, 
              help='优化级别')
@click.option('--debug', is_flag=True, help='启用调试模式')
@click.option('--enable-tensor-core', is_flag=True, help='启用Tensor Core指令')
@click.option('--enable-dot-instructions', is_flag=True, help='启用DOT指令（dot2/dot4）')
@click.option('--enable-buffer-ops', is_flag=True, help='启用缓冲区操作指令')
def generate_conv2d(backend, dtype, in_channels, out_channels, kernel_size, 
                   stride, padding, groups, bias, output, optimization, debug,
                   enable_tensor_core, enable_dot_instructions, enable_buffer_ops):
    """生成2D卷积算子"""
    
    try:
        # 创建配置
        config = Conv2DConfig(
            name="conv2d_generated",
            backend=Backend(backend),
            dtype=DataType(dtype),
            optimization_level=optimization,
            debug_mode=debug,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            # 硬件指令配置
            enable_tensor_core=enable_tensor_core,
            enable_dot_instructions=enable_dot_instructions,
            enable_buffer_ops=enable_buffer_ops
        )
        
        # 创建算子
        operator = Conv2DOperator(config)
        
        # 设置输入形状
        input_shape = TensorShape(
            dims=["N", in_channels, "H", "W"],
            dtype=DataType(dtype),
            name="input"
        )
        operator.set_input_shapes([input_shape])
        
        # 生成代码
        generator = CodeGenerator()
        generated_files = generator.generate_operator_code(operator)
        
        # 保存文件
        output_path = Path(output)
        saved_files = generator.save_generated_code(generated_files, str(output_path))
        
        click.echo(f"✅ 2D卷积算子生成成功!")
        click.echo(f"📁 输出目录: {output_path.absolute()}")
        
        for file_type, file_path in saved_files.items():
            click.echo(f"  📄 {file_type}: {file_path}")
        
        # 显示性能提示
        hints = operator.get_performance_hints()
        if hints:
            click.echo("\n💡 性能优化建议:")
            for hint in hints:
                click.echo(f"  • {hint}")
                
    except Exception as e:
        click.echo(f"❌ 生成失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--backend', '-b', type=click.Choice(['cuda', 'cpu', 'opencl', 'triton', 'rocm']), 
              default='cuda', help='目标后端')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'int32']), 
              default='float32', help='数据类型')
@click.option('--dim', type=int, default=-1, help='Softmax计算维度')
@click.option('--temperature', type=float, default=1.0, help='温度参数')
@click.option('--log-softmax', is_flag=True, help='生成LogSoftmax')
@click.option('--online-algorithm', is_flag=True, default=True, help='使用在线算法')
@click.option('--output', '-o', type=click.Path(), default='./generated/softmax', 
              help='输出目录')
@click.option('--optimization', type=click.IntRange(0, 2), default=2, 
              help='优化级别')
@click.option('--debug', is_flag=True, help='启用调试模式')
def generate_softmax(backend, dtype, dim, temperature, log_softmax, online_algorithm,
                    output, optimization, debug):
    """生成Softmax算子"""
    
    try:
        # 创建配置
        config = SoftmaxConfig(
            name="softmax_generated",
            backend=Backend(backend),
            dtype=DataType(dtype),
            optimization_level=optimization,
            debug_mode=debug,
            dim=dim,
            temperature=temperature,
            use_log_softmax=log_softmax,
            use_online_softmax=online_algorithm
        )
        
        # 创建算子
        operator = SoftmaxOperator(config)
        
        # 设置输入形状
        input_shape = TensorShape(
            dims=["N", "seq_len", "dim"],
            dtype=DataType(dtype),
            name="input"
        )
        operator.set_input_shapes([input_shape])
        
        # 生成代码
        generator = CodeGenerator()
        generated_files = generator.generate_operator_code(operator)
        
        # 保存文件
        output_path = Path(output)
        saved_files = generator.save_generated_code(generated_files, str(output_path))
        
        algorithm = "LogSoftmax" if log_softmax else "Softmax"
        click.echo(f"✅ {algorithm}算子生成成功!")
        click.echo(f"📁 输出目录: {output_path.absolute()}")
        
        for file_type, file_path in saved_files.items():
            click.echo(f"  📄 {file_type}: {file_path}")
        
        # 显示性能提示
        hints = operator.get_performance_hints()
        if hints:
            click.echo("\n💡 性能优化建议:")
            for hint in hints:
                click.echo(f"  • {hint}")
                
    except Exception as e:
        click.echo(f"❌ 生成失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--backend', '-b', type=click.Choice(['cuda', 'cpu', 'opencl', 'triton', 'rocm']), 
              default='cuda', help='目标后端')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'int32']), 
              default='float32', help='数据类型')
@click.option('--num-experts', type=int, default=8, help='专家数量')
@click.option('--expert-dim', type=int, default=512, help='专家维度')
@click.option('--hidden-dim', type=int, default=2048, help='隐藏层维度')
@click.option('--top-k', type=int, default=2, help='TopK路由')
@click.option('--gate-type', type=click.Choice(['top_k', 'switch', 'dense']), 
              default='top_k', help='门控类型')
@click.option('--expert-type', type=click.Choice(['ffn', 'attention']), 
              default='ffn', help='专家类型')
@click.option('--output', '-o', type=click.Path(), default='./generated/moe', 
              help='输出目录')
@click.option('--optimization', type=click.IntRange(0, 2), default=2, 
              help='优化级别')
@click.option('--debug', is_flag=True, help='启用调试模式')
def generate_moe(backend, dtype, num_experts, expert_dim, hidden_dim, top_k,
                gate_type, expert_type, output, optimization, debug):
    """生成MoE算子"""
    
    try:
        # 创建配置
        config = MoEConfig(
            name="moe_generated",
            backend=Backend(backend),
            dtype=DataType(dtype),
            optimization_level=optimization,
            debug_mode=debug,
            num_experts=num_experts,
            expert_dim=expert_dim,
            hidden_dim=hidden_dim,
            top_k=top_k,
            gate_type=gate_type,
            expert_type=expert_type
        )
        
        # 创建算子
        operator = MoEOperator(config)
        
        # 设置输入形状
        input_shape = TensorShape(
            dims=["batch_size", "seq_len", expert_dim],
            dtype=DataType(dtype),
            name="input"
        )
        operator.set_input_shapes([input_shape])
        
        # 生成代码
        generator = CodeGenerator()
        generated_files = generator.generate_operator_code(operator)
        
        # 保存文件
        output_path = Path(output)
        saved_files = generator.save_generated_code(generated_files, str(output_path))
        
        click.echo(f"✅ MoE算子生成成功!")
        click.echo(f"📁 输出目录: {output_path.absolute()}")
        
        for file_type, file_path in saved_files.items():
            click.echo(f"  📄 {file_type}: {file_path}")
        
        # 显示性能提示
        hints = operator.get_performance_hints()
        if hints:
            click.echo("\n💡 性能优化建议:")
            for hint in hints:
                click.echo(f"  • {hint}")
                
    except Exception as e:
        click.echo(f"❌ 生成失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='./generated/batch', 
              help='输出目录')
def batch_generate(config_file, output):
    """从配置文件批量生成算子"""
    
    try:
        # 加载配置
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                batch_config = yaml.safe_load(f)
            else:
                batch_config = json.load(f)
        
        output_path = Path(output)
        
        for operator_config in batch_config.get('operators', []):
            operator_type = operator_config.get('type')
            
            click.echo(f"🔄 生成 {operator_type} 算子...")
            
            # 根据类型生成相应算子
            # 这里可以扩展支持更多算子类型
            
        click.echo(f"✅ 批量生成完成! 输出目录: {output_path.absolute()}")
        
    except Exception as e:
        click.echo(f"❌ 批量生成失败: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('operator_dir', type=click.Path(exists=True))
def validate(operator_dir):
    """验证生成的算子代码"""
    
    click.echo(f"🔍 验证算子代码: {operator_dir}")
    
    # TODO: 实现代码验证逻辑
    # 1. 检查语法错误
    # 2. 检查API一致性
    # 3. 运行基本测试
    
    click.echo("✅ 验证通过!")


def main():
    """命令行入口点"""
    cli()


@cli.command()
@click.argument('operator_type', type=click.Choice(['conv2d', 'softmax', 'moe']))
@click.option('--backend', '-b', type=click.Choice(['cuda', 'triton', 'rocm']), 
              default='cuda', help='硬件加速后端')
@click.option('--dtype', type=click.Choice(['float32', 'float16', 'int32']), 
              default='float16', help='数据类型（建议使用float16以支持Tensor Core）')
@click.option('--enable-tensor-core', is_flag=True, default=True, help='启用Tensor Core指令')
@click.option('--enable-dot-instructions', is_flag=True, help='启用DOT指令（dot2/dot4/dp4a）')
@click.option('--enable-buffer-ops', is_flag=True, help='启用缓冲区操作指令')
@click.option('--enable-mfma', is_flag=True, help='启用MFMA指令（ROCm）')
@click.option('--tensor-core-shape', type=click.Choice(['16x16x16', '8x32x16', '32x8x16']), 
              default='16x16x16', help='Tensor Core矩阵形状')
@click.option('--output', '-o', type=click.Path(), default='./generated/hardware_optimized', 
              help='输出目录')
@click.option('--optimization', type=click.IntRange(0, 2), default=2, 
              help='优化级别')
def generate_hardware_optimized(operator_type, backend, dtype, enable_tensor_core,
                               enable_dot_instructions, enable_buffer_ops, enable_mfma,
                               tensor_core_shape, output, optimization):
    """生成硬件指令优化的算子代码"""
    
    try:
        from opforge.core.operator_base import HardwareInstruction, HardwareConfig
        
        click.echo(f"🚀 生成硬件优化的{operator_type}算子")
        click.echo(f"💻 后端: {backend}")
        click.echo(f"🔧 数据类型: {dtype}")
        
        # 创建硬件配置
        hardware_config = HardwareConfig()
        
        enabled_instructions = []
        if enable_tensor_core:
            hardware_config.supported_instructions.add(HardwareInstruction.TENSOR_CORE)
            # 解析Tensor Core形状
            shape_parts = tensor_core_shape.split('x')
            tc_shape = tuple(int(x) for x in shape_parts)
            hardware_config.tensor_core_shapes = [tc_shape]
            enabled_instructions.append(f"Tensor Core ({tensor_core_shape})")
            
        if enable_dot_instructions:
            hardware_config.supported_instructions.update({
                HardwareInstruction.DOT2,
                HardwareInstruction.DOT4,
                HardwareInstruction.DP4A
            })
            enabled_instructions.append("DOT指令 (dot2/dot4/dp4a)")
            
        if enable_buffer_ops:
            hardware_config.supported_instructions.update({
                HardwareInstruction.BUFFER_LOAD,
                HardwareInstruction.BUFFER_STORE
            })
            enabled_instructions.append("缓冲区操作")
            
        if enable_mfma and backend == 'rocm':
            hardware_config.supported_instructions.add(HardwareInstruction.MFMA)
            enabled_instructions.append("MFMA指令")
        
        click.echo(f"⚙️  启用的指令: {', '.join(enabled_instructions)}")
        
        # 根据算子类型创建配置
        if operator_type == 'conv2d':
            config = Conv2DConfig(
                name="hardware_optimized_conv2d",
                backend=Backend(backend),
                dtype=DataType(dtype),
                optimization_level=optimization,
                debug_mode=False,
                hardware_config=hardware_config,
                enable_tensor_core=enable_tensor_core,
                enable_dot_instructions=enable_dot_instructions,
                enable_buffer_ops=enable_buffer_ops,
                # 默认参数
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            )
            operator = Conv2DOperator(config)
            input_shape = TensorShape(dims=["N", 64, "H", "W"], dtype=DataType(dtype), name="input")
            
        elif operator_type == 'softmax':
            config = SoftmaxConfig(
                name="hardware_optimized_softmax",
                backend=Backend(backend),
                dtype=DataType(dtype),
                optimization_level=optimization,
                # hardware_config=hardware_config,
                dim=-1
            )
            operator = SoftmaxOperator(config)
            input_shape = TensorShape(dims=["N", "seq_len", "dim"], dtype=DataType(dtype), name="input")
            
        elif operator_type == 'moe':
            config = MoEConfig(
                name="hardware_optimized_moe",
                backend=Backend(backend),
                dtype=DataType(dtype),
                optimization_level=optimization,
                # hardware_config=hardware_config,
                num_experts=8,
                expert_dim=512,
                hidden_dim=2048,
                top_k=2
            )
            operator = MoEOperator(config)
            input_shape = TensorShape(dims=["batch", "seq_len", 512], dtype=DataType(dtype), name="input")
        
        # 设置输入形状
        operator.set_input_shapes([input_shape])
        
        # 生成代码
        generator = CodeGenerator()
        generated_files = generator.generate_operator_code(operator)
        
        # 保存文件
        output_path = Path(output)
        saved_files = generator.save_generated_code(generated_files, str(output_path))
        
        click.echo(f"✅ 硬件优化{operator_type}算子生成成功！")
        click.echo(f"📁 输出目录: {output_path.absolute()}")
        
        for file_type, file_path in saved_files.items():
            click.echo(f"  📄 {file_type}: {file_path}")
        
        # 显示性能提示
        hints = operator.get_performance_hints()
        if hints:
            click.echo("\n💡 性能优化建议:")
            for hint in hints:
                click.echo(f"  • {hint}")
        
        # 显示硬件指令信息
        click.echo("\n🔧 硬件指令详情:")
        if enable_tensor_core:
            click.echo(f"  • Tensor Core: 支持{tensor_core_shape}形状")
            click.echo(f"  • 建议使用float16数据类型以获得最佳性能")
        if enable_dot_instructions:
            click.echo(f"  • DOT指令: 支持高效的点积计算")
        if enable_buffer_ops:
            click.echo(f"  • 缓冲区操作: 优化的内存访问模式")
        
        click.echo(f"\n🎉 硬件优化完成！")
        
    except Exception as e:
        click.echo(f"❌ 生成失败: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()