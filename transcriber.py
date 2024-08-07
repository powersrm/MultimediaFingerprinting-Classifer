import whisper
import ssl
import warnings

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module='whisper')


class Transcriber:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_audio(self, input_file, output_file):
        try:
            print(f"Transcribing {input_file}...")
            result = self.model.transcribe(input_file, fp16=False)
            with open(output_file, "w") as f:
                f.write(result["text"])
            print(f"Transcription for {input_file} saved to {output_file}")
        except Exception as e:
            print(f"Error transcribing {input_file}: {e}")
