from models.registry import ModelRegistry
from backend.preprocessor import TextPreprocessor
from backend.translator import Translator
import torch

class Predictor:
    def __init__(self):
        self.registry = ModelRegistry()
        self.preprocessor = TextPreprocessor()
        self.translator = Translator()

    def predict(self, model_name, text, src_lang='en'):
        if src_lang != 'en':
            text = self.translator.translate_text(text, src_lang, 'en')

        model = self.registry.get_model(model_name)
        cleaned_text = self.preprocessor.clean_text(text)

        if model_name == "bert":
            tokenizer = self.registry.models["bert"].tokenizer
            encoding = tokenizer(
                cleaned_text,
                add_special_tokens=True,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(model.device)
            attention_mask = encoding['attention_mask'].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
        else:
            predicted_class = model.predict([cleaned_text])[0]

        return predicted_class