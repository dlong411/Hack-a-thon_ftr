import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file. Prefers OpenAI's transcription API if OPENAI_API_KEY is set. Otherwise, falls back to local whisper if installed.

    Returns the transcript text.
    """
    # Prefer OpenAI API
    if OPENAI_API_KEY:
        try:
            import openai

            openai.api_key = OPENAI_API_KEY
            # Use the OpenAI audio transcription endpoint (model name may change)
            with open(file_path, "rb") as f:
                # This call may vary with openai sdk versions; it's a best-effort wrapper.
                transcription = openai.Audio.transcribe("whisper-1", f)
                # transcription may be a dict with 'text'
                if isinstance(transcription, dict) and transcription.get("text"):
                    return transcription["text"]
                return str(transcription)
        except Exception as e:
            print("OpenAI transcription failed, falling back to local whisper:", e)

    # Fallback to local whisper (if installed)
    try:
        import whisper

        model = whisper.load_model("small")
        result = model.transcribe(file_path)
        return result.get("text", "")
    except Exception as e:
        raise RuntimeError("No transcription method available: " + str(e))


if __name__ == "__main__":
    print("transcribe.py module - call transcribe_audio(file_path)")
