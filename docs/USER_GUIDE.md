# OpForge 使用指南

欢迎使用 OpForge - 深度学习算子自动生成工具！本指南将帮助您快速上手并充分利用 OpForge 的强大功能。

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/opforge/opforge.git
cd opforge

# 安装依赖
pip install -r requirements.txt

# 安装OpForge
pip install -e .
```

### 验证安装

```bash
opforge --version
opforge list-backends
```

## 基本用法

### 1. 使用命令行工具

OpForge 提供了友好的命令行界面，可以快速生成常用算子：

#### 生成2D卷积算子

```bash
# 基本用法
opforge generate conv2d --in-channels 64 --out-channels 128 --kernel-size 3

# 指定后端和优化选项
opforge generate conv2d \
    --backend cuda \
    --in-channels 64 \
    --out-channels 128 \
    --kernel-size 3 \
    --stride 1 \
    --padding 1 \
    --groups 1 \
    --optimization 2 \
    --output ./my_conv2d
```

#### 生成Softmax算子

```bash
# 标准Softmax
opforge generate softmax --dim -1 --backend cuda

# LogSoftmax with temperature scaling
opforge generate softmax \
    --dim -1 \
    --temperature 0.5 \
    --log-softmax \
    --online-algorithm \
    --output ./my_softmax
```

#### 生成MoE算子

```bash
# 基本MoE
opforge generate moe \
    --num-experts 8 \
    --expert-dim 512 \
    --hidden-dim 2048 \
    --top-k 2 \
    --gate-type top_k \
    --backend cuda
```

### 2. 使用Python API

对于更复杂的需求，可以直接使用Python API：

```python
from opforge.core import Backend, DataType, TensorShape
from opforge.operators import Conv2DOperator
from opforge.operators.conv_operator import Conv2DConfig
from opforge.core.code_generator import CodeGenerator

# 创建配置
config = Conv2DConfig(
    name="my_conv2d",
    backend=Backend.CUDA,
    dtype=DataType.FLOAT32,
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    optimization_level=2
)

# 创建算子
operator = Conv2DOperator(config)

# 设置输入形状
input_shape = TensorShape(
    dims=[1, 64, 224, 224],
    dtype=DataType.FLOAT32,
    name="input"
)
operator.set_input_shapes([input_shape])

# 生成代码
generator = CodeGenerator()
generated_files = generator.generate_operator_code(operator)

# 保存文件
saved_files = generator.save_generated_code(generated_files, "./output")
```

## 高级用法

### 1. 自定义模板

OpForge 使用 Jinja2 模板系统，您可以自定义代码生成模板：

```python
# 创建自定义模板目录
mkdir -p custom_templates/cuda

# 复制并修改现有模板
cp src/opforge/templates/cuda/conv2d_kernel.cu custom_templates/cuda/

# 使用自定义模板
generator = CodeGenerator(template_dir="custom_templates")
```

### 2. 批量生成

使用配置文件批量生成多个算子：

```yaml
# batch_config.yaml
operators:
  - type: conv2d
    backend: cuda
    config:
      in_channels: 64
      out_channels: 128
      kernel_size: 3
      
  - type: softmax
    backend: cuda
    config:
      dim: -1
      use_log_softmax: true
      
  - type: moe
    backend: cuda
    config:
      num_experts: 8
      expert_dim: 512
```

```bash
opforge batch-generate batch_config.yaml
```

### 3. 性能调优

#### CUDA优化选项

- **优化级别 0**: 基础实现，适合调试
- **优化级别 1**: 启用基础优化（shared memory, coalescing）
- **优化级别 2**: 激进优化（Tensor Cores, 循环展开）

#### 内存优化

```python
# 检查内存需求
memory_req = operator.get_memory_requirements()
print(f"总内存需求: {memory_req['total_memory_bytes'] / 1024**3:.2f} GB")

# 获取性能建议
hints = operator.get_performance_hints()
for hint in hints:
    print(f"建议: {hint}")
```

### 4. 扩展新算子

创建自定义算子：

```python
from opforge.core.operator_base import OperatorBase, OperatorConfig

@dataclass
class MyOperatorConfig(OperatorConfig):
    # 自定义配置参数
    param1: int = 10
    param2: float = 1.0

class MyOperator(OperatorBase):
    def get_operator_type(self) -> str:
        return "my_operator"
    
    def validate_config(self) -> bool:
        # 验证配置
        return True
    
    def infer_output_shape(self, input_shapes):
        # 推断输出形状
        return input_shapes
    
    def generate_forward_code(self) -> str:
        # 生成前向代码
        return "// My operator forward code"
    
    def generate_backward_code(self) -> str:
        # 生成反向代码
        return "// My operator backward code"
```

## 最佳实践

### 1. 算子选择

- **卷积算子**: 适用于CNN网络，支持多种卷积变体
- **Softmax算子**: 适用于注意力机制和分类任务
- **MoE算子**: 适用于大规模模型的稀疏计算

### 2. 后端选择

- **CUDA**: 最佳性能，适合训练和推理
- **CPU**: 通用性强，适合边缘设备
- **OpenCL**: 跨平台，适合移动设备

### 3. 数据类型选择

- **Float32**: 最佳精度，适合训练
- **Float16**: 平衡精度和速度，适合推理
- **Int8**: 最快速度，适合量化推理

### 4. 性能优化

1. **选择合适的优化级别**
2. **根据硬件特性调整参数**
3. **使用性能分析工具验证**
4. **关注内存访问模式**

## 故障排除

### 常见问题

#### 1. 后端不可用

```bash
# 检查可用后端
opforge list-backends

# 如果CUDA不可用，检查驱动
nvidia-smi
```

#### 2. 编译错误

```bash
# 检查CUDA工具链
nvcc --version

# 检查依赖库
pkg-config --libs cudnn
```

#### 3. 内存不足

```python
# 减少批次大小或优化内存使用
config.optimization_level = 1  # 减少优化级别
```

### 调试技巧

1. **启用调试模式**
   ```python
   config.debug_mode = True
   ```

2. **检查生成的代码**
   ```bash
   opforge validate ./generated/my_operator
   ```

3. **使用性能分析工具**
   ```bash
   nsys profile python my_operator_test.py
   ```

## 贡献指南

欢迎为 OpForge 贡献代码！请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

## 更多资源

- [API文档](docs/api/)
- [算子实现指南](docs/operators/)
- [性能优化指南](docs/performance/)
- [示例项目](examples/)
- [FAQ](docs/faq.md)

## 联系我们

- GitHub Issues: [https://github.com/opforge/opforge/issues](https://github.com/opforge/opforge/issues)
- 讨论区: [https://github.com/opforge/opforge/discussions](https://github.com/opforge/opforge/discussions)
- 邮箱: team@opforge.ai