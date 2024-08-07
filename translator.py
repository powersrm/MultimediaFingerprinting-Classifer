from deep_translator import GoogleTranslator


class Translator:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate_text(self, text):
        try:
            # Split the text into manageable chunks
            def split_text(text, max_length=5000):
                chunks = []
                current_chunk = []
                current_length = 0

                for word in text.split():
                    if current_length + len(word) + 1 > max_length:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word) + 1
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1

                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                return chunks

            # Translate chunks and combine results
            chunks = split_text(text)
            translated_chunks = [self.translator.translate(chunk) for chunk in chunks]
            translated_text = ' '.join(translated_chunks)

            return translated_text
        except Exception as e:
            print(f"Error translating text: {e}")
            return ""
