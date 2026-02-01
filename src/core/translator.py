from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, source='auto', target='zh-CN'):
        self.source = source
        self.target = target
        self.translator = GoogleTranslator(source=source, target=target)

    def set_languages(self, source, target):
        if source != self.source or target != self.target:
            self.source = source
            self.target = target
            self.translator = GoogleTranslator(source=source, target=target)

    def translate(self, text):
        if not text or not text.strip():
            return ""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Error: {e}]"
