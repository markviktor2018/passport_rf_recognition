from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "DeepPavlov/rubert-base-cased"
save_path = "/media/alex/6011659d-0cd0-4325-8e75-ac1515aeeb591/Projects/personal_files/PassportOCR/models/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float16)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Модель сохранена в {save_path}")
