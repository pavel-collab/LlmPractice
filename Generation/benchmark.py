from transformers import pipeline
import nltk
from rouge import Rouge
import numpy as np

# Загрузка необходимых ресурсов для nltk
nltk.download('punkt')

# Инициализация модели для генерации текста
generator = pipeline('text-generation', model='gpt-2')

# Референсный текст (эталонный текст)
reference_text = "The quick brown fox jumps over the lazy dog."

# Генерация текста с помощью модели
input_text = "The quick brown fox"
generated_text = generator(input_text, max_length=50, num_return_sequences=1)[0]['generated_text']

print(f"Generated Text: {generated_text}")

# Токенизация текста
generated_tokens = nltk.word_tokenize(generated_text)
reference_tokens = nltk.word_tokenize(reference_text)

# Вычисление метрики BLEU
bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], generated_tokens)
print(f"BLEU Score: {bleu_score}")

# Вычисление метрик ROUGE
rouge = Rouge()
rouge_scores = rouge.get_scores(generated_text, reference_text)
print(f"ROUGE Scores: {rouge_scores}")

# Вычисление метрики METEOR
# Для этого нужно установить nltk и загрузить необходимые ресурсы
nltk.download('wordnet')
nltk.download('omw-1.4')
meteor_score = nltk.translate.meteor_score.meteor_score([reference_tokens], generated_tokens)
print(f"METEOR Score: {meteor_score}")

# Вычисление перплексии (Perplexity)
# Для вычисления перплексии нужно использовать модель с вероятностями
# Здесь приведен пример с использованием модели GPT-2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Токенизация текста
input_ids = tokenizer.encode(generated_text, return_tensors='pt')

# Вычисление перплексии
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = torch.exp(loss)

print(f"Perplexity: {perplexity.item()}")