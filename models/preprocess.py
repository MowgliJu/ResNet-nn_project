import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

# Средние и стандартные отклонения для нормализации
mean = np.array([0.4713, 0.4699, 0.4549])
std = np.array([0.2801, 0.2687, 0.2749])

# Трансформация для тестирования и валидации
def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

# Преобразование изображения для предсказания
def preprocess_image(image_path, transform):
    """
    Загрузка изображения, применение трансформаций и возвращение тензора.
    
    Args:
        image_path (str): Путь к изображению.
        transform (torchvision.transforms.Compose): Трансформация для изображения.
    
    Returns:
        torch.Tensor: Изображение, преобразованное в тензор.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Добавление batch dimension
        return image_tensor
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке изображения: {e}")

