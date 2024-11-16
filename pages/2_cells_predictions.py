import streamlit as st
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from models.resnet18 import CellsResNet, load_model
from models.preprocess import get_transform, preprocess_image
import requests
from io import BytesIO

# Настройка устройства
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
# MODEL_PATH_1 = 'models/cells_resnet18.pth'  # Первая модель
# # Вторая модель, замените на актуальный путь
# # MODEL_PATH_2 = 'models/second_model.pth'
class_names = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE',
               2: 'MONOCYTE', 3: 'NEUTROPHIL'}

# Функция для загрузки модели


def load_model_from_path(model_path, device):
    return load_model(model_path, device)

# Функция предсказания


def predict_image_class(model, image_path, class_names, device):
    transform = get_transform()
    image_tensor = preprocess_image(image_path, transform)

    model.eval()
    with torch.inference_mode():
        start_time = time.time()
        outputs = model(image_tensor.to(device))
        _, predicted_class_idx = outputs.max(1)
        predicted_class = class_names[predicted_class_idx.item()]
        elapsed_time = time.time() - start_time

    return predicted_class, elapsed_time, image_tensor

# Визуализация результата


def display_prediction_result(predicted_class, elapsed_time, image_tensor):
    image = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.title(
        f"Предсказанный класс: {predicted_class}\nВремя предсказания: {elapsed_time:.4f} секунд")
    plt.axis("off")
    st.pyplot(plt)


# Интерфейс Streamlit
st.title("Приложение для классификации изображений")

# Выбор модели из боковой панели
# model_option = st.sidebar.selectbox(
#     "Выберите модель", ["Первая модель", "Вторая модель"])

# if model_option == "Первая модель":
#     model = load_model_from_path(MODEL_PATH_1, DEVICE)
# elif model_option == "Вторая модель":
#     model = load_model_from_path(MODEL_PATH_2, DEVICE)

# Выбор загрузки изображений
upload_method = st.radio(
    "Как вы хотите загрузить изображения?", ("По ссылке", "Файлы"))

if upload_method == "По ссылке":
    url = st.text_input("Введите ссылку на изображение")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Загруженное изображение по ссылке",
                     use_container_width=True)

            # Прогнозируем класс
            predicted_class, elapsed_time, image_tensor = predict_image_class(
                model, url, class_names, DEVICE)

            # Отображаем результат
            display_prediction_result(
                predicted_class, elapsed_time, image_tensor)
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения: {e}")
else:
    uploaded_files = st.file_uploader(
        "Выберите изображения", type="jpg", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Сохраняем изображение в файл
            image_path = f"temp_image_{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Отображаем изображение
            image = Image.open(uploaded_file)
            st.image(
                image, caption=f"Загруженное изображение: {uploaded_file.name}", use_container_width=True)

            # Прогнозируем класс
            predicted_class, elapsed_time, image_tensor = predict_image_class(
                model, image_path, class_names, DEVICE)

            # Отображаем результат
            display_prediction_result(
                predicted_class, elapsed_time, image_tensor)

# Страница статистики обучения
page = st.sidebar.radio("Выберите страницу", [
                        "Главная", "Статистика обучения"])

if page == "Статистика обучения":
    st.title("Статистика обучения модели")

    # Замените на свои данные
    st.write("Графики потерь и точности:")
    st.write("Графики метрик:")
    st.image("images/metric.png", caption="Метрики обучения",
             use_container_width=True)

    st.write("Confusion Matrix Image:")
    st.image("images/confusion_map.png",
             caption="Confusion Matrix Image", use_container_width=True)
