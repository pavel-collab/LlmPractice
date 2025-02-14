import pandas as pd
import hydra
from omegaconf import DictConfig
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data import TextDataset
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding

from datasets import Dataset

from custom_trainer import CustomTrainer

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)    

    validation_accuracy  = accuracy_score (predictions, labels)
    validation_precision = precision_score(predictions, labels)
    validation_recall    = recall_score   (predictions, labels)
    validation_f1_micro  = f1_score       (predictions, labels, average='micro')
    validation_f1_macro  = f1_score       (predictions, labels, average='macro')

    return {
        'accuracy': validation_accuracy,
        'precision': validation_precision,
        'recall': validation_recall,
        'f1_micro': validation_f1_micro,
        'f1_macro': validation_f1_macro
    }

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=128
    )

@hydra.main(config_path="./conf", config_name="config", version_base="1.2 ")
def main(cfg: DictConfig):
    # get raw data from file
    df = pd.read_csv(f"./{cfg.dataset.path_to_train_data}", sep='\t')
    labels = df['class'].tolist()
    texts = df['tweet'].tolist()

    # inspect a list of labels to make a weights of classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # get a tokenizer and model itself
    checkpoint = 'google/bert_uncased_L-2_H-128_A-2'
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    #! note that in this impemenation we pass only texts and labels.
    #! we will use tokenizer on another step
    #? where using tokenizer is better? On the step of the building dataset or during build batch like here?
    # dataset = TextDataset(texts, labels)

    #! An alternative way to build a dataset, using hugging face datasets package:
    data_dict = {'text': texts, 'label': labels}
    dataset = Dataset.from_dict(data_dict)

    # Задание процента разделения (например, 20% на валидацию)
    train_test_ratio = 0.2
    # Разделение на тренировочный и валидационный наборы
    train_dataset, eval_dataset = dataset.train_test_split(test_size=train_test_ratio).values()


    # we make a data_collator, an object that will form a correct batch from the dataset
    # it adds a padding to the sentences, it need know a tokinizer to know what kind of padding
    # token to use
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',            # директория для логов TensorBoard
        logging_steps=10
    )

    '''
    When logs will be written u will need to install tensorboard firstly

    ```
    pip install tensorboar
    ```
    
    and apply it with a log dir

    ```
    tensorboard --logdir=./logs
    ```

    Finally tensorboard will run on your localhost on the 6006 port, so the only thing u need is to go to the localhost:

    ```
    http://localhost:6006
    ```
    '''

    # Create a CustomTrainer instance
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        class_weights=class_weights
    )

    trainer.train()


if __name__ == '__main__':
    main()