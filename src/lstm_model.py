from torch import nn

class LSTMGenerator(nn.Module):
    def __init__(self, tokenizer, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.tokenizer = tokenizer

        self.embeding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask):
        x = self.embeding(input_ids)
        rnn_out, _ = self.lstm(x)

        rnn_out_normed = self.norm(rnn_out)

        mask = attention_mask.unsqueeze(2).expand_as(rnn_out_normed)
        masked_out = rnn_out_normed * mask
        summed = masked_out.sum(dim=1)
        lengths = attention_mask.sum(dim=1).unsqueeze(1)
        mean_pooled = summed / lengths

        out = self.dropout(mean_pooled)
        logits = self.fc(out)

        return logits
    
    def generate(self, input_ids, attention_mask, max_len=20):
        num_words = 0
        while num_words < max_len:
            logits = self.forward(input_ids, attention_mask)
            token = logits.argmax(dim=1)

            # input_ids = torch.cat((input_ids[0], torch.tensor([token]).to(device))).unsqueeze(dim=0)
            # attention_mask = torch.cat((attention_mask[0], torch.tensor([1]).to(device))).unsqueeze(dim=0)

            input_ids = torch.cat([input_ids, token.unsqueeze(dim=1)], dim=1)
            new_token_mask = (token != tokenizer.eos_token_id).long()
            attention_mask = torch.cat([attention_mask, new_token_mask.unsqueeze(dim=1)], dim=1)
            
            num_words += 1

            # раняя остановка, если все новые токены стали eos
            if new_token_mask.sum() == 0:
                return input_ids.to('cpu')   
                               
        return input_ids.to('cpu')