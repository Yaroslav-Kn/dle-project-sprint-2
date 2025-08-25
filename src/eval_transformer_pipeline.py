from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline
import pandas as pd

def eval_transformer(model_name: str = "distilgpt2", 
                     df_test: pd.DataFrame,
                     column_data: str = 'data',
                     column_end_text: str = 'end_text'):

    rouge = evaluate.load("rouge")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model_name, device=device, pad_token_id=tokenizer.eos_token_id)


    test_data = [tokenizer.decode(data) for data in df_test[column_data]]
    trues = [tokenizer.decode(end) for end in df_test[column_end_text]]

    result = generator(test_data, max_new_tokens=20, do_sample=True)
    preds = [res[0]["generated_text"] for res in result]

    for i, (pred, data) in enumerate(zip(preds, test_data)):
        preds[i] = pred[len(data):]

    results = rouge.compute(predictions = preds, references = trues)
    return results