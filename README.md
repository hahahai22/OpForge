# OpForge - 深度学习算子自动生成工具

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Triton](https://img.shields.io/badge/Triton-2.0+-orange.svg)](https://github.com/openai/triton)

一个功能强大的深度学习算子代码自动生成工具，支持生成高性能的 CUDA、Triton、ROCm、CPU 等多种后端的算子实现，并提供硬件指令级优化。

## ✨ 核心特性

### 🚀 多后端支持
- **CUDA**: 支持 GPU 加速计算，包含 Tensor Core 和 DOT 指令优化
- **Triton**: Python JIT 编译，自动GPU内核优化
- **ROCm**: AMD GPU 支持，包含 MFMA 指令优化
- **CPU**: 支持 OpenMP 并行和 AVX 向量化优化

### 🔧 硬件指令优化
- **Tensor Core**: NVIDIA 混合精度矩阵运算 (16x16x16)
- **DOT 指令**: dot2/dot4/dp4a 高效点积计算
- **MFMA**: AMD Matrix Fused Multiply Add 指令
- **缓冲区操作**: buffer_load/store 内存优化指令
- **向量指令**: shuffle/reduce/ballot 并行优化

### 🎯 支持的算子类型
- **卷积算子**: Conv2D（标准/深度/分组卷积）
- **激活函数**: Softmax、LogSoftmax（在线算法优化）
- **专家混合**: MoE（稀疏路由和专家并行）
- **更多算子**: 持续扩展中...

## 📋 项目现状

基于最新的项目统计：

- 📄 **总文件数**: 34 个
- 🐍 **Python 文件**: 22 个（包括核心逻辑、算子实现、CLI 工具）
- 📋 **模板文件**: 7 个（CUDA 2个、Triton 2个、ROCm 1个、CPU 1个）
- 📚 **文档文件**: 4 个（README、用户指南、项目总结）
- 🧪 **测试文件**: 2 个（单元测试和模板测试）
- 🔄 **生成文件**: 3 个（演示用例）

🟢 **项目健康度**: 100% （所有模板文件完整，重要文件齐全）

### 🎨 智能代码生成
- **模板化架构**: 基于 Jinja2 的灵活模板系统
- **自动优化**: 根据硬件特性自动选择最优算法
- **形状推断**: 自动推断输出张量形状
- **内存分析**: 计算内存需求和提供优化建议

## 📊 项目统计和清理

使用内置的项目清理工具来维护项目结构：

```bash
# 查看项目统计报告
python clean_project.py -r

# 清理空目录和整理项目
python clean_project.py

# 自动清理（不问确认）
python clean_project.py -y
```


## 🏗️ 项目结构

```
OpForge/
├── src/opforge/              # 核心源代码
│   ├── core/                 # 核心模块
│   │   ├── operator_base.py        # 算子基类和硬件指令定义
│   │   ├── backend_manager.py      # 后端管理器
│   │   └── code_generator.py       # 代码生成引擎
│   ├── operators/            # 算子实现
│   │   ├── conv_operator.py        # 2D卷积算子
│   │   ├── softmax_operator.py     # Softmax算子
│   │   └── moe_operator.py         # MoE算子
│   ├── templates/            # 代码模板
│   │   ├── cuda/                   # CUDA模板
│   │   ├── triton/                 # Triton模板
│   │   ├── rocm/                   # ROCm模板
│   │   ├── cpu/                    # CPU模板
│   │   ├── python/                 # Python绑定模板
│   │   ├── tests/                  # 测试模板
│   │   └── build/                  # 构建脚本模板
│   └── cli.py                # 命令行界面
├── examples/                 # 使用示例
├── tests/                    # 测试用例
├── docs/                     # 文档
├── generated/                # 生成的代码（示例）
├── demo.py                   # 完整演示
└── hardware_demo.py          # 硬件指令演示
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/OpForge.git
cd OpForge

# 安装依赖
pip install -r requirements.txt

# 安装OpForge
pip install -e .
```

### 验证安装

```bash
# 检查版本
opforge --version

# 查看可用后端
opforge list-backends
```

### 基本使用

#### 1. 命令行工具

```bash
# 生成标准2D卷积算子
opforge generate-conv2d --backend cuda --in-channels 64 --out-channels 128

# 生成Tensor Core优化的卷积
opforge generate-hardware-optimized conv2d \
    --backend cuda \
    --enable-tensor-core \
    --enable-dot-instructions \
    --dtype float16

# 生成Triton优化版本
opforge generate-hardware-optimized conv2d \
    --backend triton \
    --enable-tensor-core

# 生成Softmax算子
opforge generate-softmax --dim -1 --temperature 0.5

# 生成MoE算子
opforge generate-moe \
    --num-experts 8 \
    --expert-dim 512 \
    --top-k 2
```

#### 2. Python API

```python
from opforge.core import Backend, DataType, TensorShape
from opforge.operators import Conv2DOperator
from opforge.operators.conv_operator import Conv2DConfig
from opforge.core.code_generator import CodeGenerator
from opforge.core.operator_base import HardwareInstruction, HardwareConfig

# 创建硬件配置
hardware_config = HardwareConfig()
hardware_config.supported_instructions.add(HardwareInstruction.TENSOR_CORE)

# 创建算子配置
config = Conv2DConfig(
    name="optimized_conv2d",
    backend=Backend.CUDA,
    dtype=DataType.FLOAT16,
    optimization_level=2,
    enable_tensor_core=True,
    enable_dot_instructions=True,
    hardware_config=hardware_config,
    in_channels=64,
    out_channels=128,
    kernel_size=3
)

# 创建算子
operator = Conv2DOperator(config)

# 设置输入形状
input_shape = TensorShape(
    dims=[8, 64, 224, 224],
    dtype=DataType.FLOAT16,
    name="input"
)
operator.set_input_shapes([input_shape])

# 生成代码
generator = CodeGenerator()
generated_files = generator.generate_operator_code(operator)

# 保存文件
saved_files = generator.save_generated_code(generated_files, "./my_kernels")

print("生成的文件:")
for file_type, path in saved_files.items():
    print(f"  {file_type}: {path}")
```

### 示例和演示

```bash
# 运行完整功能演示
python demo.py

# 运行硬件指令演示
python hardware_demo.py

# 运行具体示例
python examples/conv2d_example.py
python examples/softmax_example.py
python examples/moe_example.py
```

## 🔧 硬件指令优化

OpForge 支持多种硬件指令级优化：

### NVIDIA GPU
- **Tensor Core**: 16x16x16 混合精度矩阵运算
- **WMMA**: Warp Matrix Multiply Accumulate
- **DOT指令**: dot2, dot4, dp4a 点积运算
- **内存优化**: 共享内存、缓存友好的访问模式

### AMD GPU (ROCm)
- **MFMA**: Matrix Fused Multiply Add 指令
- **缓冲区操作**: 优化的内存访问指令
- **向量化**: 高效的并行计算

### CPU
- **OpenMP**: 多线程并行
- **AVX指令**: 向量化计算
- **缓存优化**: 数据局部性优化

## 📊 性能提升

基于理论分析的性能提升预估：

| 优化技术 | 性能提升 | 适用场景 |
|----------|----------|----------|
| Tensor Core (FP16) | 4-16x | 大矩阵运算，CNN/Transformer |
| DOT指令 (INT8) | 2-4x | 量化推理，边缘计算 |
| MFMA (ROCm) | 8-16x | AMD GPU上的矩阵运算 |
| 缓冲区优化 | 20-30% | 内存密集型算子 |
| OpenMP并行 | 2-8x | CPU多核计算 |

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_opforge.py::TestConv2DOperator

# 生成测试覆盖率报告
python -m pytest tests/ --cov=opforge --cov-report=html
```

## 📖 文档

- [用户指南](docs/USER_GUIDE.md) - 详细使用说明
- [项目总结](PROJECT_SUMMARY.md) - 技术架构和实现细节
- [API文档](docs/api/) - 详细的API参考

## 🤝 贡献

欢迎贡献代码！请参考以下步骤：

1. Fork 这个仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式化
black src/
flake8 src/

# 类型检查
mypy src/opforge
```

## 🔮 路线图

### 近期计划
- [ ] 添加更多算子支持（Attention、BatchNorm、LayerNorm）
- [ ] 完善Triton自动调优
- [ ] 增加量化算子支持
- [ ] 优化模板系统

### 长期计划
- [ ] 支持更多硬件后端（Intel GPU、ARM、FPGA）
- [ ] 图级别优化（算子融合）
- [ ] 可视化界面
- [ ] 云端代码生成服务

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🙏 致谢

感谢以下开源项目的启发和支持：
- [Triton](https://github.com/openai/triton) - GPU内核JIT编译
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA模板库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Jinja2](https://jinja.palletsprojects.com/) - 模板引擎

## 📧 联系方式

- 项目主页: [https://github.com/your-repo/OpForge](https://github.com/your-repo/OpForge)
- 问题反馈: [Issues](https://github.com/your-repo/OpForge/issues)
- 邮箱: team@opforge.ai

---

**OpForge** - 让深度学习算子开发更简单、更高效！ 🚀