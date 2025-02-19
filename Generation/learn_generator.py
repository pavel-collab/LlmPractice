
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset

# 1. Загрузка токенизатора и модели
model_name = "gpt2"  # Или другая предобученная модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Подготовка данных
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=True
    )

# Загрузите ваш набор данных
train_dataset = load_dataset("math_texts.txt", tokenizer)

# 3. Настройка аргументов для тренировки
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# 4. Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 5. Обучение модели
trainer.train()

# 6. Сохранение дообученной модели
trainer.save_model("./math_gpt2")
