import streamlit as st
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from models.resnet18 import MyFreezeResNet, load_model
from models.preprocess import get_transform, preprocess_image
from models.cell_resnet18 import CellsResNet, load_model1
from models.cell_preprocessing import get_transform1, preprocess_image1
import requests
from io import BytesIO

# Настройка устройства
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к моделям
MODEL_PATH_1 = 'models/my_freeze_resnet18.pth'  # Первая модель
MODEL_PATH_2 = 'models/cells_resnet18.pth'  # Вторая модель
class_names = {
    0: 'air hockey', 1: 'ampute football', 2: 'archery', 3: 'arm wrestling', 
    4: 'axe throwing', 5: 'balance beam', 6: 'barell racing', 7: 'baseball',
    8: 'basketball', 9: 'baton twirling', 10: 'bike polo', 11: 'billiards',
    12: 'bmx', 13: 'bobsled', 14: 'bowling', 15: 'boxing', 16: 'bull riding',
    17: 'bungee jumping', 18: 'canoe slamon', 19: 'cheerleading', 20: 'chuckwagon racing',
    21: 'cricket', 22: 'croquet', 23: 'curling', 24: 'disc golf', 25: 'fencing',
    26: 'field hockey', 27: 'figure skating men', 28: 'figure skating pairs',
    29: 'figure skating women', 30: 'fly fishing', 31: 'football', 32: 'formula 1 racing',
    33: 'frisbee', 34: 'gaga', 35: 'giant slalom', 36: 'golf', 37: 'hammer throw',
    38: 'hang gliding', 39: 'harness racing', 40: 'high jump', 41: 'hockey',
    42: 'horse jumping', 43: 'horse racing', 44: 'horseshoe pitching', 45: 'hurdles',
    46: 'hydroplane racing', 47: 'ice climbing', 48: 'ice yachting', 49: 'jai alai',
    50: 'javelin', 51: 'jousting', 52: 'judo', 53: 'lacrosse', 54: 'log rolling',
    55: 'luge', 56: 'motorcycle racing', 57: 'mushing', 58: 'nascar racing',
    59: 'olympic wrestling', 60: 'parallel bar', 61: 'pole climbing', 62: 'pole dancing',
    63: 'pole vault', 64: 'polo', 65: 'pommel horse', 66: 'rings', 67: 'rock climbing',
    68: 'roller derby', 69: 'rollerblade racing', 70: 'rowing', 71: 'rugby',
    72: 'sailboat racing', 73: 'shot put', 74: 'shuffleboard', 75: 'sidecar racing',
    76: 'ski jumping', 77: 'sky surfing', 78: 'skydiving', 79: 'snow boarding',
    80: 'snowmobile racing', 81: 'speed skating', 82: 'steer wrestling', 83: 'sumo wrestling',
    84: 'surfing', 85: 'swimming', 86: 'table tennis', 87: 'tennis', 88: 'track bicycle',
    89: 'trapeze', 90: 'tug of war', 91: 'ultimate', 92: 'uneven bars', 93: 'volleyball',
    94: 'water cycling', 95: 'water polo', 96: 'weightlifting', 97: 'wheelchair basketball',
    98: 'wheelchair racing', 99: 'wingsuit flying'
}
class_names1 = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}

# Функция для загрузки модели
def load_model_from_path(model_path, model_type, device):
    if model_type == "Первая модель":
        return load_model(model_path, device)
    elif model_type == "Вторая модель":
        return load_model1(model_path, device)

# Функция предсказания для первой модели
def predict_image_class_first(model, image_path, class_names, device):
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

# Функция предсказания для второй модели
def predict_image_class_second(model, image_path, class_names, device):
    transform = get_transform1()
    image_tensor = preprocess_image1(image_path, transform)

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
    plt.title(f"Предсказанный класс: {predicted_class}\nВремя предсказания: {elapsed_time:.4f} секунд")
    plt.axis("off")
    st.pyplot(plt)

# Интерфейс Streamlit
st.title("Приложение для классификации изображений")

# Выбор модели из боковой панели
model_option = st.sidebar.selectbox("Выберите модель", ["Первая модель", "Вторая модель"])

if model_option == "Первая модель":
    model = load_model_from_path(MODEL_PATH_1, model_option, DEVICE)
    current_class_names = class_names
    predict_function = predict_image_class_first
elif model_option == "Вторая модель":
    model = load_model_from_path(MODEL_PATH_2, model_option, DEVICE)
    current_class_names = class_names1
    predict_function = predict_image_class_second

# Выбор загрузки изображений
upload_method = st.radio("Как вы хотите загрузить изображения?", ("По ссылке", "Файлы"))

if upload_method == "По ссылке":
    url = st.text_input("Введите ссылку на изображение")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Загруженное изображение по ссылке", use_container_width=True)

            # Прогнозируем класс
            predicted_class, elapsed_time, image_tensor = predict_function(model, url, current_class_names, DEVICE)

            # Отображаем результат
            display_prediction_result(predicted_class, elapsed_time, image_tensor)
        except Exception as e:
            st.error(f"Ошибка при загрузке изображения: {e}")
else:
    uploaded_files = st.file_uploader("Выберите изображения", type="jpg", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Сохраняем изображение в файл
            image_path = f"temp_image_{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Отображаем изображение
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Загруженное изображение: {uploaded_file.name}", use_container_width=True)

            # Прогнозируем класс
            predicted_class, elapsed_time, image_tensor = predict_function(model, image_path, current_class_names, DEVICE)

            # Отображаем результат
            display_prediction_result(predicted_class, elapsed_time, image_tensor)

# Страница статистики обучения
page = st.sidebar.radio("Выберите страницу", ["Главная", "Статистика обучения"])

if page == "Статистика обучения":
    st.title("Статистика обучения моделей")

    # Выбор модели
    model_option = st.selectbox(
        "Выберите модель:",
        ["Модель 1", "Модель 2"],
    )

    # Данные для первой модели
    if model_option == "Модель 1":
        st.write("### Статистика для Модели 1")
        st.write("F1 Score: 0.8708")
        st.write("Графики потерь и точности:")
        st.image("images/metric.png", caption="Метрики обучения Модели 1", use_container_width=True)
        st.write("Confusion Matrix:")
        st.image("images/confusion_map.png", caption="Матрица ошибок Модели 1", use_container_width=True)

    # Данные для второй модели
    elif model_option == "Модель 2":
        st.write("### Статистика для Модели 2")
        st.write("F1 Score: 0.8211")  # Исправленный F1 Score
        st.write("Графики потерь и точности:")
        st.image("images/mod2.png", caption="Метрики обучения Модели 2", use_container_width=True)
        st.write("Confusion Matrix:")
        st.image("images/conf_matrix.jpg", caption="Матрица ошибок Модели 2", use_container_width=True)
