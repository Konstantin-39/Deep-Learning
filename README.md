# Нейронные сети для криптографии и генерации текста
==========================

Проект по разработке двух нейронных сетей:
1. Декодировщик шифра Цезаря на основе RNN
2. Генератор текста в стиле Симпсонов с использованием кастомной RNN-ячейки

## Описание проекта
--------------

Проект состоит из двух основных заданий:

### Задание 1: Расшифровка шифра Цезаря
* Реализация алгоритма шифрования сдвигом
* Создание RNN-сети для декодирования
* Обучение на наборе зашифрованных фраз
* Точность расшифровки достигает 98.6%

### Задание 2: Генерация текста
* Разработка кастомной RNN-ячейки
* Обучение на диалогах персонажей Симпсонов
* Генерация новых фраз в том же стиле

## Технические детали
-------------------

### Используемые технологии
* PyTorch для построения и обучения моделей
* Pandas для обработки данных
* CUDA для ускорения вычислений

### Архитектура решений

#### Шифр Цезаря
```python
class CaesarDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super(CaesarDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.shift_embedding = nn.Embedding(11, hidden_size)
        self.rnn = nn.RNN(hidden_size * 2, hidden_size, batch_first=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
```

#### Кастомная RNN-ячейка
```python
class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
```

## Установка и запуск
---------------------

### Необходимые библиотеки
```bash
pip install torch torchvision pandas
```

### Команды для запуска
```bash
# Для обучения модели шифра Цезаря
python train_caesar.py

# Для обучения генератора текста
python train_generator.py
```

## Примеры использования
----------------------

### Расшифровка текста
```python
text = "hello world"
shift = random.randint(1, 10)
encrypted_text = caesar_cipher(text, shift)
decoded_text = decode_text(model, encrypted_text, shift)
print(f"Original: {text}")
print(f"Encrypted: {encrypted_text}")
print(f"Decoded: {decoded_text}")
```

### Генерация текста
```python
start_text = "homer"
generated_text = generate_text(model, start_text, max_length=50)
print(f"Generated text: {generated_text}")
```

## Результаты
-------------

### Модель шифра Цезаря
* Loss: 0.0271
* Accuracy: 98.63%
* Успешное декодирование текстов с различными сдвигами

### Генератор текста
* Loss: 1.1595
* Успешная генерация текста в стиле диалогов Симпсонов