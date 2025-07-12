import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x_res + attn_output
        x = x + self.ff(self.norm2(x))
        return x

class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, max_length=192):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.output(x)

    def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200, top_k=50):
            """
            Генерирует ответ на основе промпта.
            
            При авторегрессии контекст сдвигается на 1 токен влево:
            - Изначально: [prompt_tokens]
            - После первого предсказания: [prompt_tokens, predicted_token]
            - При следующем предсказании: [prompt_tokens[1:], predicted_token, new_prediction]
            - И так далее, пока не достигнем max_length или EOS
            """
            self.eval()
            with torch.no_grad():
                input_ids = self.tokenizer.encode(prompt).ids
                device = next(self.parameters()).device
                input_ids = torch.tensor([input_ids], device=device)
                generated = input_ids.clone()

                for _ in range(max_out_tokens):
                    outputs = self(input_ids)
                    logits = outputs[0, -1, :] / temperature

                    if top_k is not None:
                        topk_logits, topk_indices = torch.topk(logits, top_k)
                        probs = torch.zeros_like(logits)
                        probs[topk_indices] = torch.softmax(topk_logits, dim=-1)
                    else:
                        probs = torch.softmax(logits, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    next_token = next_token.unsqueeze(0)

                    generated = torch.cat([generated, next_token], dim=1)
                    input_ids = generated[:, -context_len:]
                    if next_token.item() == self.eos_token_id:
                        break

            return self.tokenizer.decode(generated[0].tolist())


    def generate_beam_search(self, prompt, context_len=50, max_out_tokens=200, beam_width=5, eos_token_id=None):
        """
        Генерация текста с помощью beam search.
        beam_width - количество гипотез, которые сохраняются на каждом шаге.
        eos_token_id - ID токена конца последовательности.
        Возвращает наиболее вероятную сгенерированную последовательность.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], device=device)

            sequences = [(input_ids, 0.0)]  # список кортежей

            for _ in range(max_out_tokens):
                all_candidates = []

                for seq, score in sequences:
                    if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                        # Если EOS достигнут сохраняю без изменений
                        all_candidates.append((seq, score))
                        continue

                    input_seq = seq[:, -context_len:]
                    outputs = self(input_seq)
                    logits = outputs[0, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # top beam_width вероятных токенов
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                    for i in range(beam_width):
                        next_token = top_indices[i].unsqueeze(0).unsqueeze(0)  # shape (1,1)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + top_log_probs[i].item()
                        all_candidates.append((new_seq, new_score))

                # Сортируем все кандидаты по убыванию логарифма вероятности и берем top beam_width
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

                # Если все последовательности достигли EOS прерываем генерацию
                if all(seq[0, -1].item() == eos_token_id for seq, _ in sequences):
                    break

            # Берем лучшую по вероятности последовательность
            best_seq = sequences[0][0][0].tolist()

            return self.tokenizer.decode(best_seq)
