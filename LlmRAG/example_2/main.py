from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration 
from transformers import Trainer, TrainingArguments 
from datasets import load_dataset 
import torch 
import faiss 
import numpy as np

# Загрузка модели RAG

model_name = "facebook/rag-token-nq" 
tokenizer = RagTokenizer.from_pretrained(model_name) 
retriever = RagRetriever.from_pretrained(model_name, index_name="exact") 
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

# Загрузка данных из Википедии

dataset = load_dataset("wikipedia", "20220301.simple") 
passages = dataset["train"]["text"][:10000]  # Используем 10 000 статей для примера

# Создание индекса FAISS

d = 768  # Размерность эмбеддингов 
index = faiss.IndexFlatL2(d) 
embeddings = np.random.rand(len(passages), d).astype('float32')  # Здесь должен быть реальный эмбеддинг index.add(embeddings)

# Подготовка обучающего датасета

def tokenize_function(examples): 
    inputs = examples["title"]  
    # Используем заголовки статей как вопросы 
    return tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Настройки обучения

training_args = TrainingArguments(output_dir="./rag_model", 
                                  evaluation_strategy="epoch", 
                                  per_device_train_batch_size=2, 
                                  per_device_eval_batch_size=2, 
                                  save_steps=500, 
                                  save_total_limit=2, 
                                  num_train_epochs=3, 
                                  logging_dir="./logs")

trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=tokenized_datasets["train"])

# Запуск обучения

trainer.train()

# Сохранение модели

model.save_pretrained("./rag_trained_model") 
tokenizer.save_pretrained("./rag_trained_model")

# Сохранение индекса FAISS

faiss.write_index(index, "faiss_index.idx")
