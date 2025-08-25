import re
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import os

def preprocessing_data(path_row_data: str = 'data/raw_dataset.txt',
                       path_processed_data: str = 'data/processed.csv',
                       path_train_data: str = 'data/train.csv',
                       path_valid_data: str = 'data/valid.csv',
                       path_test_data: str = 'data/test.csv',
                       model_name: str = "distilgpt2",
                       random_seed: int = 123,):
    
    if os.path.exists(path_row_data):
        with open(path_row_data, 'r', encoding='utf-8') as file:
            texts = file.readlines()
    else:
        raise ValueError(f'Отсутствует файл с сырыми данными {path_row_data}')

    for file_name in [
                       path_processed_data,
                       path_train_data,
                       path_valid_data,
                       path_test_data
    ]:
        dir = os.path.dirname(file_name)
        if not os.path.exists(dir):
            os.makedirs(dir) 

    texts = [clean_text(text) for text in tqdm(texts, desc='Очистка тестка')]
    
    word_counts = [len(text.split(' ')) for text in texts]
    mask = [count < 35 for count in word_counts]
    texts = [item for item, mask_value in zip(texts, mask) if mask_value]
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_texts = [tokenizer(text, truncation=True)['input_ids'] for text in tqdm(texts, desc='Очистка тестка')]

    data_list, target_list, end_list = get_data_and_target(token_texts)

    df = pd.DataFrame({
        'data': data_list,
        'target': target_list,
        'end_text': end_list
    })

    df.to_csv(path_processed_data, index=False)

    df_train, df_test= train_test_split(df, test_size=0.2, random_state=random_seed)
    df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train.to_csv(path_train_data, index=False)
    df_valid.to_csv(path_valid_data, index=False)
    df_test.to_csv(path_test_data, index=False)




def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\w*@\w*', ' ', text) # убираем указани аккаунта и почты
    text = re.sub(r'http\S*', ' ', text) # убираем адреса

    text = re.sub(r"[:;=]['\-]?[)d(p]+", ' ', text) # убираем классические смайлики
    text = re.sub(r'\*\w*\*', ' ', text) # убираем смайлики прописанные в формате слов
    # убираем самйлики в формате кодов
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
    text = regrex_pattern.sub(r'',text)

    text = re.sub(r'\s+([.,!?;:])', r'\1', text) # убираем висячие знаки припинания
    re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text) # добавляем пробелы после знаков припинания, если их нет

    # заменяем все возможные пробелы, переносы и т.д. на стандартный символ пробела и затем заменяем 2 и более пробелов на 1
    text = re.sub(r'_x000d_\n', ' ', text)
    text = re.sub(r'^\s+|\n', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub(r'^ +| +$', '', text) # убираем пробелы в начале и конце строки
    return text

def get_data_and_target(token_texts: list[str]) -> tuple[list[str], list[str]]:
    set_tuple = set()
    data_list = []
    target_list = []
    end_list = []
    for text in tqdm(token_texts, desc='Создание списков для обучения'):
        start_point = int(len(text) / 4 * 3) # По заданию для генерации используем 3/4 текста
        for idx_target in range(max(1, start_point), len(text)):
            # Добавляем только уникальные сочетания data/target чтобы избежать дублей
            item = ' '.join([str(i) for i in text[:idx_target + 1]])
            if (item) not in set_tuple:
                set_tuple.add(item)
                data_list.append(text[:idx_target])
                target_list.append(text[idx_target])
                end_list.append(text[idx_target:])
    return data_list, target_list, end_list