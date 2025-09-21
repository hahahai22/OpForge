#!/usr/bin/env python3
"""
OpForge 项目清理脚本
自动清理空文件夹，整理项目结构，并提供项目统计信息
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Set


class ProjectCleaner:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.removed_dirs: List[str] = []
        self.file_stats: Dict[str, int] = {}
        self.empty_dirs: Set[str] = set()
        
    def find_empty_directories(self) -> List[Path]:
        """查找所有空目录"""
        empty_dirs = []
        
        # 排除的目录（git相关目录等）
        exclude_patterns = {
            '.git', '__pycache__', '.pytest_cache', 
            'node_modules', '.venv', 'venv', 'env',
            '.mypy_cache', 'dist', 'build', '*.egg-info'
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # 过滤掉排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_patterns 
                      and not d.endswith('.egg-info')]
            
            current_dir = Path(root)
            
            # 检查目录是否为空（没有文件且没有非空子目录）
            if not files:
                # 检查子目录是否都是空的
                has_content = False
                for subdir in dirs:
                    subdir_path = current_dir / subdir
                    if self._has_content(subdir_path):
                        has_content = True
                        break
                
                if not has_content and current_dir != self.project_root:
                    empty_dirs.append(current_dir)
                    
        return empty_dirs
    
    def _has_content(self, directory: Path) -> bool:
        """递归检查目录是否有内容"""
        try:
            for item in directory.iterdir():
                if item.is_file():
                    return True
                elif item.is_dir() and self._has_content(item):
                    return True
            return False
        except (PermissionError, OSError):
            return True  # 如果无法访问，假设有内容
    
    def collect_file_statistics(self) -> Dict[str, int]:
        """收集文件统计信息"""
        stats = {
            'total_files': 0,
            'python_files': 0,
            'template_files': 0,
            'config_files': 0,
            'doc_files': 0,
            'test_files': 0,
            'generated_files': 0,
            'cuda_files': 0,
            'triton_files': 0,
            'rocm_files': 0,
            'cpu_files': 0
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # 排除某些目录
            dirs[:] = [d for d in dirs if not d.startswith('.') 
                      and d != '__pycache__' and not d.endswith('.egg-info')]
            
            for file in files:
                if file.startswith('.') or file.endswith('.pyc'):
                    continue
                    
                stats['total_files'] += 1
                file_path = Path(root) / file
                path_str = str(file_path).lower()
                
                # 根据文件扩展名和路径分类
                if file.endswith('.py'):
                    stats['python_files'] += 1
                    if 'test' in file.lower() or 'test' in path_str:
                        stats['test_files'] += 1
                    elif 'triton' in path_str:
                        stats['triton_files'] += 1
                elif file.endswith('.cu'):
                    stats['template_files'] += 1
                    stats['cuda_files'] += 1
                elif file.endswith(('.cpp', '.c')) and 'rocm' in path_str:
                    stats['template_files'] += 1
                    stats['rocm_files'] += 1
                elif file.endswith(('.cpp', '.c')):
                    stats['template_files'] += 1
                    stats['cpu_files'] += 1
                elif file.endswith(('.h', '.cmake')):
                    stats['template_files'] += 1
                elif file.endswith(('.md', '.rst', '.txt')):
                    stats['doc_files'] += 1
                elif file.endswith(('.json', '.yaml', '.yml', '.toml', '.ini')):
                    stats['config_files'] += 1
                
                # 检查是否是生成的文件
                if 'generated' in path_str:
                    stats['generated_files'] += 1
        
        return stats
    
    def clean_empty_directories(self, confirm: bool = True) -> int:
        """清理空目录"""
        empty_dirs = self.find_empty_directories()
        
        if not empty_dirs:
            print("✅ 没有发现空目录")
            return 0
        
        print(f"🔍 发现 {len(empty_dirs)} 个空目录:")
        for dir_path in empty_dirs:
            relative_path = dir_path.relative_to(self.project_root)
            print(f"  - {relative_path}")
        
        if confirm:
            response = input("\n是否删除这些空目录? (y/N): ")
            if response.lower() != 'y':
                print("❌ 取消删除操作")
                return 0
        
        removed_count = 0
        for dir_path in empty_dirs:
            try:
                dir_path.rmdir()
                relative_path = dir_path.relative_to(self.project_root)
                self.removed_dirs.append(str(relative_path))
                removed_count += 1
                print(f"🗑️  删除空目录: {relative_path}")
            except OSError as e:
                print(f"❌ 无法删除 {dir_path}: {e}")
        
        return removed_count
    
    def check_missing_files(self) -> List[str]:
        """检查可能缺失的重要文件"""
        important_files = [
            'README.md',
            'requirements.txt',
            'setup.py',
            'LICENSE',
            '.gitignore'
        ]
        
        missing = []
        for file in important_files:
            if not (self.project_root / file).exists():
                missing.append(file)
        
        return missing
    
    def check_template_completeness(self) -> Dict[str, List[str]]:
        """检查模板文件完整性"""
        template_dir = self.project_root / 'src' / 'opforge' / 'templates'
        
        expected_backends = ['cuda', 'triton', 'rocm', 'cpu', 'python', 'tests', 'build']
        expected_files = {
            'cuda': ['conv2d_kernel.cu', 'conv2d_kernel.h'],
            'triton': ['conv2d_kernel.py'],
            'rocm': ['conv2d_kernel.cpp'],
            'cpu': ['conv2d_kernel.cpp'],
            'python': ['conv2d_binding.py'],
            'tests': ['conv2d_test.py'],
            'build': ['cuda_build.cmake']
        }
        
        missing_templates = {}
        
        for backend in expected_backends:
            backend_dir = template_dir / backend
            if not backend_dir.exists():
                missing_templates[backend] = ['整个目录缺失']
                continue
                
            missing_files = []
            for expected_file in expected_files.get(backend, []):
                if not (backend_dir / expected_file).exists():
                    missing_files.append(expected_file)
            
            if missing_files:
                missing_templates[backend] = missing_files
        
        return missing_templates
    
    def generate_project_report(self) -> str:
        """生成项目报告"""
        stats = self.collect_file_statistics()
        missing_files = self.check_missing_files()
        missing_templates = self.check_template_completeness()
        
        report = []
        report.append("📊 OpForge 项目统计报告")
        report.append("=" * 50)
        report.append(f"📁 项目根目录: {self.project_root}")
        report.append("\n📈 文件统计:")
        report.append(f"  📄 总文件数: {stats['total_files']}")
        report.append(f"  🐍 Python文件: {stats['python_files']}")
        report.append(f"  📋 模板文件: {stats['template_files']}")
        report.append(f"    ├─ CUDA文件: {stats['cuda_files']}")
        report.append(f"    ├─ Triton文件: {stats['triton_files']}")
        report.append(f"    ├─ ROCm文件: {stats['rocm_files']}")
        report.append(f"    └─ CPU文件: {stats['cpu_files']}")
        report.append(f"  📚 文档文件: {stats['doc_files']}")
        report.append(f"  🧪 测试文件: {stats['test_files']}")
        report.append(f"  ⚙️  配置文件: {stats['config_files']}")
        report.append(f"  🔄 生成文件: {stats['generated_files']}")
        
        if self.removed_dirs:
            report.append("\n🗑️  已删除的空目录:")
            for dir_name in self.removed_dirs:
                report.append(f"  - {dir_name}")
        
        if missing_files:
            report.append("\n⚠️  可能缺失的重要文件:")
            for file in missing_files:
                report.append(f"  - {file}")
        
        if missing_templates:
            report.append("\n🔧 模板文件检查:")
            for backend, files in missing_templates.items():
                report.append(f"  ❌ {backend}: 缺失 {', '.join(files)}")
        else:
            report.append("\n✅ 所有模板文件完整")
        
        # 项目结构健康度评估
        health_score = 100
        if missing_files:
            health_score -= len(missing_files) * 10
        if missing_templates:
            health_score -= len(missing_templates) * 15
        
        health_emoji = "🟢" if health_score >= 90 else "🟡" if health_score >= 70 else "🔴"
        report.append(f"\n{health_emoji} 项目健康度: {health_score}%")
        
        return "\n".join(report)
    
    def run_cleanup(self, auto_confirm: bool = False):
        """运行完整的清理流程"""
        print("🚀 开始 OpForge 项目清理...")
        print(f"📁 项目目录: {self.project_root}")
        print()
        
        # 清理空目录
        removed_count = self.clean_empty_directories(confirm=not auto_confirm)
        
        # 生成并显示报告
        print("\n" + self.generate_project_report())
        
        if removed_count > 0:
            print(f"\n✅ 清理完成！删除了 {removed_count} 个空目录")
        else:
            print("\n✅ 项目结构良好，无需清理")
        
        # 提供优化建议
        self._print_optimization_suggestions()
    
    def _print_optimization_suggestions(self):
        """打印优化建议"""
        print("\n💡 优化建议:")
        print("  1. 定期运行此脚本清理项目")
        print("  2. 确保所有模板文件完整")
        print("  3. 保持代码和文档同步")
        print("  4. 运行测试确保代码质量: python -m pytest tests/")
        print("  5. 更新依赖: pip install -r requirements.txt")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OpForge 项目清理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python clean_project.py                    # 交互式清理
  python clean_project.py -y                 # 自动确认清理
  python clean_project.py -r                 # 仅生成报告
  python clean_project.py -p /path/to/project # 指定项目路径
        """
    )
    parser.add_argument('--project-root', '-p', 
                       default='.', 
                       help='项目根目录路径 (默认: 当前目录)')
    parser.add_argument('--auto-confirm', '-y', 
                       action='store_true',
                       help='自动确认所有操作')
    parser.add_argument('--report-only', '-r',
                       action='store_true', 
                       help='仅生成报告，不删除文件')
    
    args = parser.parse_args()
    
    # 验证项目根目录
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"❌ 项目目录不存在: {project_root}")
        sys.exit(1)
    
    # 检查是否是 OpForge 项目
    opforge_markers = [
        project_root / 'src' / 'opforge',
        project_root / 'setup.py'
    ]
    
    if not any(marker.exists() for marker in opforge_markers):
        print(f"⚠️  警告: {project_root} 可能不是 OpForge 项目目录")
        if not args.auto_confirm:
            response = input("是否继续? (y/N): ")
            if response.lower() != 'y':
                sys.exit(0)
    
    # 创建清理器并运行
    cleaner = ProjectCleaner(str(project_root))
    
    if args.report_only:
        print(cleaner.generate_project_report())
    else:
        cleaner.run_cleanup(auto_confirm=args.auto_confirm)


if __name__ == '__main__':
    main()