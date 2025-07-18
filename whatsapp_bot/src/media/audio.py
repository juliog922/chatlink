import logging
import os
import wave
import tempfile
import subprocess
from vosk import Model, KaldiRecognizer
from number_parser import parser
import re


def convert_spoken_numbers(text: str) -> str:
    """
    Convierte expresiones numéricas escritas en palabras a cifras.
    Usa number-parser para identificar números compuestos.

    Args:
        text: Transcripción de Vosk en español (en minúsculas)

    Returns:
        Texto con los números normalizados (e.g. 'dos mil cinco' -> '2005')
    """
    try:
        # number-parser trabaja mejor si hay puntuación artificial
        text = re.sub(r"[^a-záéíóúñü\s]", "", text.lower())
        parsed = parser.parse(text, language="es")
        return parsed
    except Exception as e:
        logging.warning(f"Error normalizando números: {e}")
        return text


# Initialize once globally
vosk_model = Model("vosk_model_es")


def transcribe_audio(audio_bytes: bytes, extension=".ogg") -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_input_path = tmp_file.name

        wav_path = tmp_input_path + ".wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        wf = wave.open(wav_path, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() != 16000
        ):
            raise ValueError("Invalid audio format")

        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
        transcript = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                res = recognizer.Result()
                transcript += eval(res)["text"] + " "

        transcript += eval(recognizer.FinalResult())["text"]
        wf.close()

        os.remove(tmp_input_path)
        os.remove(wav_path)
        return convert_spoken_numbers(transcript.strip())

    except Exception as e:
        logging.error(f"Audio transcription error: {e}")
        return ""
