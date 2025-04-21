# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости (включая swig для faiss-cpu)
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
ENV VENV_PATH="/opt/venv"
RUN python -m venv --copies $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Копируем файлы в контейнер
WORKDIR /app
COPY requirements.txt .
COPY main.py .

# Обновляем pip и ставим зависимости
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Команда по умолчанию
CMD ["python", "main.py"]
