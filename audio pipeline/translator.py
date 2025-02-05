# filepath: backend/translator.py
from transformers import MarianMTModel, MarianTokenizer
from config import Config

class Translator:
    def __init__(self):
        self.tokenizer = MarianTokenizer.from_pretrained(Config.TRANSLATION_MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(Config.TRANSLATION_MODEL_NAME)

    def translate_text(self, text, src_lang='es', tgt_lang='en'):
        translated = self.model.generate(**self.tokenizer.prepare_seq2seq_batch([text], src_lang=src_lang, tgt_lang=tgt_lang))
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)