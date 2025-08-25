import evaluate
import torch

def eval(model, tokenizer, dataloader):
    rouge = evaluate.load("rouge")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()
    preds, trues = [], []
    for batch in dataloader:
        data = batch['data'].to(device)
        masks = batch['masks'].to(device)
        end_text = batch['end_text']

        with torch.no_grad():
            output = model.generate(data, masks)
            preds += output
            trues += end_text

    preds = [tokenizer.decode(pred) for pred in preds]
    trues = [tokenizer.decode(true) for true in trues]

    results = rouge.compute(predictions = preds, references = trues)

    return results