import os
import numpy as np
from PIL import Image  # Pillow 库用于图像处理

def to_one_hot(labels, num_classes=10):
    """
    将整数标签转换为 one-hot 编码.

    Args:
        labels (np.ndarray): 形状为 (batch_size,) 的整数标签数组.
        num_classes (int): 类别数量，默认为 10 (MNIST).

    Returns:
        np.ndarray: 形状为 (batch_size, num_classes) 的 one-hot 编码标签数组.
    """
    batch_size = labels.shape[0]
    one_hot_labels = np.zeros((batch_size, num_classes), dtype=np.float32) # 初始化为 0
    one_hot_labels[np.arange(batch_size), labels] = 1 # 设置对应类别索引为 1
    return one_hot_labels

def load_mnist_data(image_dir, label_file, flatten_images=True, one_hot_encode_labels=True): # 添加 one_hot_encode_labels 参数，默认为 True
    """
    简化版加载 MNIST 图片和标签数据 (假设 PNG 图片，标签文件索引递增)，并可选择展平图像和 one-hot 编码标签.

    Args:
        image_dir (str): 图片所在的目录路径 (例如 'mnist_dataset/train_images').
        label_file (str): 标签文件的路径 (例如 'mnist_dataset/train_labels.txt').
        flatten_images (bool, optional): 是否将图像展平成一维向量 (784). 默认为 True.
        one_hot_encode_labels (bool, optional): 是否将标签转换为 one-hot 编码 (形状为 (num_samples, num_classes)). 默认为 True.

    Returns:
        tuple: 包含两个 NumPy 数组的元组: (images, labels).
               - images:  如果 flatten_images=True, 形状为 (num_samples, 784) 的 NumPy 数组.
                         如果 flatten_images=False, 形状为 (num_samples, image_height, image_width, channels) 的 NumPy 数组 (灰度图, channels=1).
               - labels:  如果 one_hot_encode_labels=True, 形状为 (num_samples, num_classes) 的 one-hot 编码标签数组.
                         如果 one_hot_encode_labels=False, 形状为 (num_samples,) 的整数标签数组.
    """
    images = []
    labels_int = [] #  使用 labels_int 存储整数标签，后续再进行 one-hot 编码
    label_map = {} # 使用字典来存储索引和标签的映射

    # 读取标签文件
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_index = parts[0] # 假设第一个是图像索引
                label = int(parts[1])   # 第二个是标签，转换为整数
                label_map[image_index] = label

    # 遍历图像目录，加载图片并匹配标签
    image_names = sorted(os.listdir(image_dir)) # 确保文件名顺序一致，如果标签文件没有索引
    loaded_images = []
    # loaded_labels = []  不再直接使用 loaded_labels，而是添加到 labels_int

    for image_name in image_names:
        if not image_name.lower().endswith('.png'): # 简化：只检查 PNG
            continue # 跳过非 PNG 文件

        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path) # 使用 Pillow 加载图像
        image_gray = image.convert('L') # 转换为灰度图
        image_np = np.array(image_gray).astype('float32') / 255.0 # 归一化

        if flatten_images: # 如果需要展平图像
            image_np_flattened = image_np.flatten() # 展平成一维向量 (784,)
            loaded_images.append(image_np_flattened)
        else: # 否则，保持 2D 结构并添加通道维度 (如果需要 CNN 等模型)
            image_np_expanded = np.expand_dims(image_np, axis=-1) # 添加通道维度
            loaded_images.append(image_np_expanded)


        # 标签匹配 (假设文件名可以关联到标签文件中的索引)
        image_base_name = os.path.splitext(image_name)[0] # 去掉 .png 扩展名
        if image_base_name in label_map:
            label = label_map[image_base_name]
            labels_int.append(label) # 添加到 labels_int 列表
        else:
            print(f"警告: 找不到图像 {image_name} 的标签。请检查标签文件和文件名。")
            # 你可以选择跳过，或者抛出错误
            pass

    # 转换为 NumPy 数组
    images_np = np.array(loaded_images)
    labels_np_int = np.array(labels_int) # 将整数标签列表转换为 NumPy 数组

    # One-hot 编码 (如果 one_hot_encode_labels 为 True)
    if one_hot_encode_labels:
        labels_np = to_one_hot(labels_np_int, num_classes=10) # 使用 to_one_hot 函数进行编码
    else:
        labels_np = labels_np_int # 否则，直接使用整数标签

    print(f"加载完成: 图片形状 {images_np.shape}, 标签形状 {labels_np.shape}")
    return images_np, labels_np