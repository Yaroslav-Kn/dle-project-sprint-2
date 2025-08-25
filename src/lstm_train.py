from .lstm_model import LSTMGenerator
from torch.utils.data import DataLoader
from torch import optim

def train_epoch(model: LSTMGenerator, loader: DataLoader, optimizer, criterion) -> float:
    train_loss = 0
    model.train()
    for batch in loader:
        data = batch['data'].to(device)
        masks = batch['masks'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(data, masks)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss
    
    return train_loss / len(loader)

def eval(model: LSTMGenerator, loader: DataLoader, criterion) -> float:
    valid_loss = 0
    preds, trues = [], []
    model.eval()

    for batch in loader:
        data = batch['data'].to(device)
        masks = batch['masks'].to(device)
        target = batch['target'].to(device)

        with torch.no_grad():
            output = model(data, masks)
            loss = criterion(output, target)
            valid_loss += loss

            preds += torch.argmax(output.squeeze(1), dim=1).cpu().tolist()
            trues += target.tolist()

    preds = [tokenizer.decode(pred) for pred in preds]
    trues = [tokenizer.decode(true) for true in trues]

    return valid_loss / len(loader), rouge.compute(predictions = preds, references = trues)['rouge1']

def train(model: LSTMGenerator, 
          train_dataloader: DataLoader, 
          valid_dataloader: DataLoader, 
          optimizer, 
          criterion, 
          path_model = 'best_model.pth',
          pation = 10, 
          max_epoch=200)    
    rouge = evaluate.load("rouge")
    best_val = np.inf
    train_losses = []
    valid_losses = []
    metrics = []

    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(max_epoch):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        valid_loss, rouge1 = eval(model, valid_dataloader, criterion)

        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())
        metrics.append(rouge1)

        if valid_loss < best_val:
            best_val = valid_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), path_model)

        else:
            no_improve_epochs += 1
            if no_improve_epochs > pation:
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}")
            print(f"Epoch {epoch+1}: Valid loss = {valid_loss:.4f}, Rouge1 = {rouge1:.4f}") 

    print('-' * 20)
    print(f'Best epoch: {best_epoch + 1}')
    print(f'Best validation loss: {best_val:.4f}')
    print(f'Best metric rouge1: {metrics[best_epoch]:.4f}')
    return train_losses, valid_losses, metrics, best_val 