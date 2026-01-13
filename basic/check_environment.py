#!/usr/bin/env python
"""
环境检查脚本

验证所有必要的依赖和 GPU 是否正确配置
"""

import sys
import platform


def print_section(title):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_python():
    """检查 Python 版本"""
    print_section("Python Version")

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.platform()}")

    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False


def check_dali():
    """检查 DALI 安装"""
    print_section("NVIDIA DALI")

    try:
        import nvidia.dali as dali
        print(f"✓ DALI installed: {dali.__version__}")

        # 检查 GPU 支持
        try:
            import nvidia.dali.backend as dali_backend
            gpu_count = dali_backend.GetPropertyNames()
            print(f"✓ DALI with GPU support")
        except:
            print("⚠ DALI without GPU support")

        return True
    except ImportError as e:
        print(f"✗ DALI not installed: {e}")
        print("  Install with: pip install nvidia-dali-cuda120")
        return False


def check_pytorch():
    """检查 PyTorch 安装"""
    print_section("PyTorch")

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA not available")

        return True
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False


def check_gpu():
    """检查 GPU 和 CUDA"""
    print_section("GPU & CUDA")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

            # 测试 GPU 访问
            x = torch.randn(100, 100).cuda()
            y = x + x
            print(f"✓ GPU computation works")
            return True
        else:
            print("✗ No GPU detected")
            return False
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def check_dependencies():
    """检查其他依赖"""
    print_section("Dependencies")

    dependencies = {
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'minio': 'MinIO',
        'pandas': 'Pandas',
    }

    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False

    return all_ok


def check_dali_operators():
    """检查关键 DALI 操作符"""
    print_section("DALI Operators")

    try:
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types

        operators = [
            'readers.file',
            'decoders.image',
            'resize',
            'random_resized_crop',
            'flip',
            'brightness_contrast',
            'normalize',
            'external_source',
        ]

        all_ok = True
        for op_name in operators:
            try:
                parts = op_name.split('.')
                obj = fn
                for part in parts:
                    obj = getattr(obj, part)
                print(f"✓ fn.{op_name}")
            except AttributeError:
                print(f"✗ fn.{op_name} not available")
                all_ok = False

        return all_ok
    except Exception as e:
        print(f"✗ Error checking operators: {e}")
        return False


def run_simple_pipeline():
    """运行简单的测试 Pipeline"""
    print_section("Simple Pipeline Test")

    try:
        import nvidia.dali as dali
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import numpy as np
        import tempfile
        import os
        from PIL import Image

        # 创建临时测试数据
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试图像
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(os.path.join(tmpdir, f"test_{i}.jpg"))

            # 定义简单 Pipeline
            @pipeline_def
            def test_pipeline(data_dir):
                images, labels = fn.readers.file(file_root=data_dir)
                images = fn.decoders.image(images, device="mixed")
                images = fn.resize(images, size=64)
                return images, labels

            # 构建和运行
            pipe = test_pipeline(data_dir=tmpdir, batch_size=2, num_threads=1, device_id=0)
            pipe.build()
            outputs = pipe.run()

            print(f"✓ Simple pipeline executed successfully")
            print(f"  Output shape: {outputs[0].shape()}")
            return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  DALI Tutorial Environment Check")
    print("="*60)

    results = {
        'Python': check_python(),
        'DALI': check_dali(),
        'PyTorch': check_pytorch(),
        'GPU': check_gpu(),
        'Dependencies': check_dependencies(),
        'Operators': check_dali_operators(),
        'Pipeline': run_simple_pipeline(),
    }

    # 总结
    print_section("Summary")

    all_ok = all(results.values())

    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"{status} {name}")

    print()

    if all_ok:
        print("🎉 All checks passed! Ready to start DALI tutorials.")
        print("\n👉 Next step: python basic/01_hello_dali.py")
        return 0
    else:
        print("⚠️  Some checks failed. Please install missing dependencies.")
        print("\nInstall all requirements:")
        print("  pip install -r basic/requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
