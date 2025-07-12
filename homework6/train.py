import torch
from torch.utils.data import DataLoader
from models import GeneratorTransformer
from dataset import TextDataset
from tokenizers import Tokenizer
import os
from tqdm import tqdm

def train():
    """
    Обучает модель Transformer для генерации текста на заданном датасете.
    Загружает текст, токенизирует и формирует датасет с последовательностями фиксированной длины.
    Проводит обучение модели с использованием кросс-энтропийной потери и оптимизатора Adam.
    Сохраняет обученную модель в файл после завершения всех эпох.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer.from_file("mistral_tokenizer_with_specials.json")
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    dataset = TextDataset(text, tokenizer, block_size=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GeneratorTransformer(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

        for i, (x, y) in progress:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            out = out.view(-1, out.size(-1))
            y = y.view(-1)

            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/checkpoint.pt")
    print("Model saved to checkpoints/checkpoint.pt")

if __name__ == "__main__":
    train()
