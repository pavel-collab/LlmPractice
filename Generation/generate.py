
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Загрузка токенизатора и модели
model_name = "gpt2"  # Вы можете использовать другую модель для генерации текста
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Функция для генерации синтетических данных
def generate_synthetic_data(prompt, num_samples=100, max_length=50):
    synthetic_texts = []
    for _ in range(num_samples):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        synthetic_texts.append(generated_text)
    return synthetic_texts

# 3. Генерация синтетических данных
prompt = "This is a text about machine learning. It discusses"
synthetic_data = generate_synthetic_data(prompt, num_samples=100)

# 4. Создание меток для синтетических данных (например, случайные метки)
labels = [random.randint(0, 1) for _ in range(len(synthetic_data))]  # бинарная классификация

# 5. Создание DataFrame
data = pd.DataFrame({"text": synthetic_data, "label": labels})

# 6. Сохранение данных в CSV (по желанию)
data.to_csv("synthetic_data.csv", index=False)

print(data.head())
