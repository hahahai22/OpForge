"""
OpForge - 深度学习算子自动生成工具

一个强大的代码生成框架，支持自动生成高性能的深度学习算子实现。
"""

__version__ = "0.1.0"
__author__ = "OpForge Team"
__email__ = "team@opforge.ai"

from .core.operator_base import OperatorBase
from .core.code_generator import CodeGenerator
from .core.backend_manager import BackendManager

__all__ = [
    "OperatorBase",
    "CodeGenerator", 
    "BackendManager",
]