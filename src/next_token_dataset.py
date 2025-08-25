import torch
import pandas as  pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 35
MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
PAD_TOKEN = tokenizer.pad_token_id


class Custom_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df['data']
        self.target = df['target']
        self.end_text = df['end_text']

    def __len__(self) -> int:
        return len(self.target)
    
    def __getitem__(self, index):
        return {
            'data': torch.tensor(self.data[index], dtype=torch.long),
            'target': torch.tensor(self.target[index], dtype=torch.long),
            'end_text': torch.tensor(self.end_text[index], dtype=torch.long),
        }
    
def collate_fn(batch):
    data = [i['data'] for i in batch]
    target = torch.stack([i['target'] for i in batch])
    end_text = [i['end_text'] for i in batch]
    max_len = max([len(i) for i in data])
    max_len = min([MAX_LEN, max_len])

    pad_text = pad_sequence(data, batch_first=True, padding_value=PAD_TOKEN)

    masks = (pad_text != 0).long()

    return {
        'data': pad_text,
        'target': target,
        'end_text': end_text,
        'masks': masks
    }   