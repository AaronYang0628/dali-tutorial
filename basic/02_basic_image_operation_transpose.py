"""
演示 HWC vs CHW 转换及其意义
"""

import numpy as np

try:
    import torch
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def demo_format_difference():
    """演示 HWC 和 CHW 的区别"""
    print("\n" + "="*60)
    print("Demo 1: HWC vs CHW 数据排列")
    print("="*60)

    # 创建一个简单的 3×3 RGB 图像
    image_hwc = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # 第一行：红、绿、蓝
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]],  # 第二行：黄、品红、青
        [[128, 128, 128], [0, 0, 0], [255, 255, 255]]  # 第三行：灰、黑、白
    ], dtype=np.uint8)

    print(f"\nHWC 格式 (3, 3, 3):")
    print(f"形状: {image_hwc.shape}")
    print(f"第一个像素 (0,0): RGB = {image_hwc[0, 0]}")
    print(f"第二个像素 (0,1): RGB = {image_hwc[0, 1]}")
    print(f"\n完整数据:")
    print(image_hwc)

    # 转换为 CHW
    image_chw = np.transpose(image_hwc, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    print(f"\n{'='*60}")
    print(f"CHW 格式 (3, 3, 3):")
    print(f"形状: {image_chw.shape}")
    print(f"\n红色通道 (完整 3×3 矩阵):")
    print(image_chw[0])
    print(f"\n绿色通道 (完整 3×3 矩阵):")
    print(image_chw[1])
    print(f"\n蓝色通道 (完整 3×3 矩阵):")
    print(image_chw[2])


def demo_pytorch_requirement():
    """演示 PyTorch 模型对输入格式的要求"""
    print("\n" + "="*60)
    print("Demo 2: PyTorch 模型输入格式要求")
    print("="*60)

    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch 未安装，跳过此 Demo")
        print("   但原理同样适用于所有使用 NCHW 格式的框架")
        return

    # 加载预训练模型
    print("\n加载 ResNet18 模型...")
    model = models.resnet18(weights=None)
    model.eval()

    # 正确的 CHW 格式
    input_chw = torch.randn(1, 3, 224, 224)  # NCHW: (Batch, Channel, H, W)
    print(f"\n✅ 正确格式 (NCHW): {input_chw.shape}")
    print(f"   - Batch: {input_chw.shape[0]}")
    print(f"   - Channels: {input_chw.shape[1]}")
    print(f"   - Height: {input_chw.shape[2]}")
    print(f"   - Width: {input_chw.shape[3]}")

    with torch.no_grad():
        output = model(input_chw)
    print(f"   - 输出形状: {output.shape}")

    # 错误的 HWC 格式
    input_hwc = torch.randn(1, 224, 224, 3)  # NHWC: (Batch, H, W, Channel)
    print(f"\n❌ 错误格式 (NHWC): {input_hwc.shape}")
    print(f"   - 如果传入模型会报错！")

    # 转换 HWC -> CHW
    input_fixed = input_hwc.permute(0, 3, 1, 2)  # NHWC -> NCHW
    print(f"\n🔧 转换后 (NCHW): {input_fixed.shape}")
    with torch.no_grad():
        output = model(input_fixed)
    print(f"   - 现在可以正常运行了！")
    print(f"   - 输出形状: {output.shape}")


def demo_dali_integration():
    """演示 DALI 如何为 PyTorch 准备数据"""
    print("\n" + "="*60)
    print("Demo 3: DALI 为 PyTorch 准备数据")
    print("="*60)

    print("\nDALI Pipeline 处理流程:")
    print("1. 读取图像 (JPEG 文件)")
    print("2. 解码 → HWC 格式 (224, 224, 3)")
    print("3. Resize/Crop → 仍然是 HWC")
    print("4. Normalize → 仍然是 HWC")
    print("5. fn.transpose(perm=[2, 0, 1]) → CHW 格式 (3, 224, 224)")
    print("6. 输出到 PyTorch → 可以直接使用 ✅")

    print("\n代码示例:")
    print("""
@pipeline_def
def pytorch_pipeline(file_list):
    images, labels = fn.readers.file(files=file_list)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, size=224)
    images = fn.cast(images, dtype=types.FLOAT)
    images = images / 255.0

    # 关键步骤：HWC -> CHW
    images = fn.transpose(images, perm=[2, 0, 1])  # (H, W, C) -> (C, H, W)

    return images, labels

# 输出直接可以喂给 PyTorch 模型
for images, labels in dali_loader:
    output = model(images)  # ✅ 直接使用，无需转换
    """)


def demo_memory_layout():
    """演示内存布局的差异"""
    print("\n" + "="*60)
    print("Demo 4: 内存布局和性能影响")
    print("="*60)

    # 创建大图像
    image_hwc = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_chw = np.transpose(image_hwc, (2, 0, 1))

    print(f"\nHWC 格式:")
    print(f"  形状: {image_hwc.shape}")
    print(f"  内存布局: RGBRGBRGB... (像素交错)")
    print(f"  第一个像素的 RGB: {image_hwc[0, 0]}")
    print(f"  访问所有 R 值: 需要跨步访问 ❌")

    print(f"\nCHW 格式:")
    print(f"  形状: {image_chw.shape}")
    print(f"  内存布局: RRRR...GGGG...BBBB... (通道连续)")
    print(f"  第一个像素位置的值: R={image_chw[0, 0, 0]}, G={image_chw[1, 0, 0]}, B={image_chw[2, 0, 0]}")
    print(f"  访问所有 R 值: 连续内存访问 ✅")

    print(f"\n卷积操作时:")
    print(f"  CHW: GPU 可以连续读取整个通道 → 高效")
    print(f"  HWC: GPU 需要跨步读取每个通道 → 低效")


def demo_framework_comparison():
    """对比不同框架的格式"""
    print("\n" + "="*60)
    print("Demo 5: 不同框架的格式偏好")
    print("="*60)

    print("\n框架格式对比:")
    print("┌─────────────────┬──────────┬─────────────┬──────────────────┐")
    print("│ 框架            │ 默认格式 │ Batch 格式  │ 示例形状         │")
    print("├─────────────────┼──────────┼─────────────┼──────────────────┤")
    print("│ PyTorch         │ CHW      │ NCHW        │ (32, 3, 224, 224)│")
    print("│ TensorFlow      │ HWC      │ NHWC        │ (32, 224, 224, 3)│")
    print("│ ONNX Runtime    │ CHW      │ NCHW        │ (32, 3, 224, 224)│")
    print("│ OpenCV/PIL      │ HWC      │ -           │ (224, 224, 3)    │")
    print("│ NumPy (一般)    │ HWC      │ -           │ (224, 224, 3)    │")
    print("└─────────────────┴──────────┴─────────────┴──────────────────┘")

    print("\n实践建议:")
    print("1. DALI → PyTorch: 使用 fn.transpose 转为 CHW ✅")
    print("2. DALI → TensorFlow: 保持 HWC 格式 ✅")
    print("3. OpenCV 读图 → PyTorch: 需要 transpose ✅")
    print("4. PIL 读图 → PyTorch: 需要 transpose ✅")


if __name__ == "__main__":
    demo_format_difference()
    demo_pytorch_requirement()
    demo_dali_integration()
    demo_memory_layout()
    demo_framework_comparison()

    if not TORCH_AVAILABLE:
        print("\n📝 提示: 安装 PyTorch 来运行完整的 Demo 2")
        print("   pip install torch torchvision")

    print("\n" + "="*60)
    print("✓ Demo completed!")
    print("="*60)
    print("\n核心要点:")
    print("1. HWC: 像素连续，OpenCV/PIL/TensorFlow 使用")
    print("2. CHW: 通道连续，PyTorch/ONNX 使用")
    print("3. fn.transpose 用于适配不同框架的输入要求")
    print("4. CHW 格式对 GPU 卷积运算更高效")
