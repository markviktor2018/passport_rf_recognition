import re
from typing import Dict, Optional
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import json

# Логи по умолчанию — инфо, чтобы видеть ход работы (и промпты, и ошибки).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self, model_path: str = "UrukHan/t5-russian-spell"):
        # Проверяем, есть ли GPU, иначе всё будет на CPU (и это сильно медленнее).
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        try:
            # Загружаем токенизатор и модель T5 для исправления OCR ошибок.
            # Эта модель заточена под русскую орфографию.
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()  # в режим инференса, без градиентов
            
        except Exception as e:
            # Если что-то не так с загрузкой — сразу валим с ошибкой.
            logger.error(f"Ошибка инициализации модели из {model_path}: {e}")
            raise

    def build_prompt(self, fields: Dict[str, str]) -> str:
        """Формируем текстовый промпт для модели LLM"""
        logger.info("Формирование промпта для LLM")
        
        # Собираем только нужные поля (ФИО, даты, серия, код подразделения и т.д.)
        fields_str = "\n".join([f"{field}: {value}" for field, value in fields.items() 
                               if field in ("name", "sername", "fam", "birth_date", "birth_place", 
                                           "series_number", "issued_code", "issued_by", "sex", "mrz")])
        
        # Тут руками прописаны форматы, чтобы модель понимала «правильный ответ»
        # и старалась приводить к этому виду.
        prompt = (
            "Ты эксперт по российским паспортам. Исправь ошибки OCR в следующих данных. "
            "Учти стандартные форматы и взаимосвязи между полями.\n\n"
            "Форматы:\n"
            "- series_number: 4 цифры, пробел, 2 цифры, пробел, 4 цифры (например: '1234 56 7890')\n"
            "- issued_code: 3 цифры, дефис, 3 цифры (например: '123-456')\n"
            "- birth_date: ДД.ММ.ГГГГ (например: '01.01.1990')\n"
            "- sex: 'МУЖ.' или 'ЖЕН.'\n\n"
            "Исправь следующие данные и верни ТОЛЬКО валидный JSON без каких-либо пояснений:\n\n"
            f"{fields_str}\n\n"
            "Исправленный JSON:"
        )
        return prompt

    def verify_all_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """
        Прогоняем все поля через LLM, которая должна «починить» формат и ошибки OCR.
        Если модель что-то не так вернёт — подстраховываемся пост-валидацией.
        """
        prompt = self.build_prompt(fields)
        logger.info(f"Промпт: {prompt}")
        
        try:
            # Токенизируем вход и гоняем через модель.
            input_ids = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).input_ids.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=512,
                    num_beams=5,              # beam search, чтобы результат был стабильнее
                    repetition_penalty=2.0,   # штраф за повторы
                    do_sample=False,          # детерминированный выход
                    temperature=0.1,          # фактически игнорируется при do_sample=False
                    early_stopping=True,
                    no_repeat_ngram_size=3    # убирает «заикания» в тексте
                )
            
            result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"Сырой ответ от модели: {result}")
            
            # Чистим всякий мусор из текста
            cleaned_result = self.clean_model_output(result)
            
            # Пытаемся сразу распарсить как JSON
            try:
                corrected_fields = json.loads(cleaned_result)
                
                # Если модель вернула не dict (например, список) — fallback
                if not isinstance(corrected_fields, dict):
                    logger.error("Модель вернула не словарь")
                    return self.post_validate_fields(fields)
                    
            except json.JSONDecodeError as e:
                # JSON кривой? Попробуем выдрать фигурные скобки руками.
                logger.error(f"Ошибка парсинга JSON: {e}")
                json_match = re.search(r'\{[^{}]*\}', cleaned_result)
                if json_match:
                    try:
                        corrected_fields = json.loads(json_match.group())
                    except:
                        logger.error("Не удалось извлечь JSON из ответа")
                        return self.post_validate_fields(fields)
                else:
                    logger.error("JSON не найден в ответе модели")
                    return self.post_validate_fields(fields)

            # На всякий случай — пробежимся своими проверками.
            corrected_fields = self.post_validate_fields(corrected_fields)
            logger.info(f"Исправленные поля: {corrected_fields}")
            
            # Если модель что-то пропустила — добавляем из исходных данных.
            for field in fields:
                if field not in corrected_fields:
                    corrected_fields[field] = fields[field]
                    
            return corrected_fields

        except Exception as e:
            # Любая ошибка при работе с моделью — не дропаем всё,
            # просто возвращаем поля после примитивной валидации.
            logger.error(f"Ошибка при обработке полей: {e}")
            return self.post_validate_fields(fields)
    
    def clean_model_output(self, text: str) -> str:
        """Очищаем строку от лишнего, чтобы json.loads не падал"""
        text = text.strip().strip('"').strip(',').strip()
        text = text.replace("'", '"')  # JSON только с двойными кавычками
        text = re.sub(r'\s+', ' ', text)
        # Иногда модель выдает `ключ: значение` без кавычек — добавляем их.
        text = re.sub(r'(\w+):\s*([^",{]+)(?=[,}])', r'\1: "\2"', text)
        return text

    def post_validate_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """Финальная зачистка: форматируем серии, даты, код подразделения и т.д."""
        corrected = fields.copy()

        # Проверка серии-номера паспорта
        if "series_number" in corrected:
            digits = re.sub(r"[^0-9]", "", corrected["series_number"])
            if len(digits) == 10:
                corrected["series_number"] = f"{digits[:4]} {digits[4:6]} {digits[6:10]}"
            elif len(digits) > 10:
                corrected["series_number"] = f"{digits[:4]} {digits[4:6]} {digits[6:10]}"
            elif len(digits) >= 4:
                corrected["series_number"] = digits[:10]

        # Проверка кода подразделения
        if "issued_code" in corrected:
            digits = re.sub(r"[^0-9]", "", corrected["issued_code"])
            if len(digits) == 6:
                corrected["issued_code"] = f"{digits[:3]}-{digits[3:6]}"
            else:
                match = re.search(r"(\d{3})\D*?(\d{3})", corrected["issued_code"])
                if match:
                    corrected["issued_code"] = f"{match.group(1)}-{match.group(2)}"

        # Приведение дат к нормальному формату
        date_patterns = [
            r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})",
            r"(\d{1,2})\s*[а-яА-Я]+\s*(\d{4})",
        ]
        
        for field in ["birth_date", "issue_date"]:
            if field in corrected:
                for pattern in date_patterns:
                    match = re.search(pattern, corrected[field])
                    if match:
                        parts = match.groups()
                        if len(parts) == 3:
                            d, m, y = parts
                        elif len(parts) == 2:
                            d, y = parts
                            m = "01"
                        else:
                            continue
                        try:
                            d, m, y = int(d), int(m) if m.isdigit() else 1, int(y)
                            if d > 31: d = 1
                            if m > 12: m = 1
                            if len(str(y)) == 2:
                                y = 1900 + y if y > 30 else 2000 + y
                            if not (1900 <= y <= 2025):
                                y = 2000
                            corrected[field] = f"{d:02d}.{m:02d}.{y:04d}"
                            break
                        except ValueError:
                            continue

        # Приведение значения пола
        if "sex" in corrected:
            text = corrected["sex"].upper().replace(" ", "")
            if re.search(r"^(МУЖ|М|M|MUJ|MALE)", text):
                corrected["sex"] = "МУЖ."
            elif re.search(r"^(ЖЕН|Ж|F|ZHEN|FEMALE)", text):
                corrected["sex"] = "ЖЕН."
            elif "М" in text or "M" in text:
                corrected["sex"] = "МУЖ."
            elif "Ж" in text or "F" in text:
                corrected["sex"] = "ЖЕН."

        # MRZ должен быть ровно 44 символа, начинаться с "P<"
        if "mrz" in corrected:
            mrz = re.sub(r"\s", "", corrected["mrz"]).upper()
            if len(mrz) == 44 and mrz.startswith("P<"):
                corrected["mrz"] = mrz

        # Для ФИО, места рождения и выдавшего органа делаем первую букву заглавной.
        for field in ["fam", "name", "sername", "birth_place", "issued_by"]:
            if field in corrected and corrected[field]:
                words = []
                for word in corrected[field].split():
                    if word and any(c.isalpha() for c in word):
                        words.append(word[0].upper() + word[1:].lower())
                    else:
                        words.append(word)
                corrected[field] = " ".join(words)

        return corrected

