import torch
from models import GeneratorTransformer
from tokenizers import Tokenizer

def chat():
    """
    Запускает интерактивный чат с моделью генерации текста на базе Transformer.
    Пользователь вводит текст, модель генерирует ответ в авторегрессивном режиме.
    Чат продолжается до ввода команды 'quit'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = Tokenizer.from_file("mistral_tokenizer_with_specials.json")
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    model = GeneratorTransformer(vocab_size=vocab_size).to(device)
    model.tokenizer = tokenizer
    model.eos_token_id = tokenizer.token_to_id("<eos>")
    model.load_state_dict(torch.load("checkpoints/checkpoint.pt", map_location=device))
    model.eval()

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break

        response = model.generate_beam_search(
            prompt=user_input,
            context_len=50,
            max_out_tokens=100,
            beam_width=15,
            eos_token_id=model.eos_token_id
        )
        print(f"Бот: {response}")

if __name__ == "__main__":
    chat()
