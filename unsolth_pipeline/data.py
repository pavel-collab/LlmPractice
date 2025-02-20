import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import random

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_length, usolth_usage=False, tokenizer=None):
        self.texts = texts
        self.labels = labels
        if not usolth_usage:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif tokenizer != None:
            self.tokenizer = tokenizer
        else:
            raise RuntimeError
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизация текста
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
class TextDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset) - int(len(self.dataset)*0.8)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    # def sample_random_item(self):
    #     dataloader = self.val_dataloader()
    #     random_batch_num = len(dataloader)
        
    #     while random_batch_num != 0:
    #         random_batch_num -= 1
    #         random_batch = next(iter(dataloader))

    #     random_sentence_idx = random.randint(0, self.batch_size)
    #     random_item = {'input_ids': random_batch['input_ids'][random_sentence_idx], 
    #                    'attention_mask': random_batch['attention_mask'][random_sentence_idx], 
    #                    'label': int(random_batch['label'][random_sentence_idx])}

    #     return random_item
    
    def tokenize_data(self, example):
        return self.tokenizer(example["sentence"],
                              truncation=True, # отвечает за обрезание слишком длинного предложения
                              padding="max_length", # отвечает за дополнение слишком короткого предложения
                              max_length=self.max_length)