from langdetect import detect
from google_trans_new import google_translator  


class Translator():

    def __init__(self, lang_tgt):
        self.lang_tgt = lang_tgt

    def translate_sentence(self, sentence):
        """ Take a sentence and get the language
            and the english translation"""
        lang_src = detect(sentence)
        if lang_src == self.lang_tgt:
            return lang_src, sentence.strip()
        else:
            translator = google_translator()
            translate_sentence = translator.translate(sentence,lang_src=lang_src,lang_tgt=self.lang_tgt)
            return lang_src, translate_sentence.strip()


if __name__ == '__main__':

    t = Translator("en")
    sentence = t.translate_sentence("Hola este es un repositorio de Github")
    print(sentence)
