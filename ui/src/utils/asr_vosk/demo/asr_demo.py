import wave
from vosk import Model, KaldiRecognizer

def transcribe_wav(wav_file_path, model_path):
    # Load the Vosk model
    model = Model(model_path)

    # Open the WAV file
    wf = wave.open(wav_file_path, "rb")

    # Check if the WAV file is in the correct format
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return

    # Create a Vosk recognizer
    recognizer = KaldiRecognizer(model, wf.getframerate())

    # Process the audio file
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result += recognizer.Result()

    # Get the final result
    result += recognizer.FinalResult()

    return result

def main():
    # Set the paths
    wav_file_path = "/raid/vladimir_albrekht/web_demo/asr_tts/demo_tts/output.wav"  # Replace with your WAV file path
    model_path = "/raid/vladimir_albrekht/web_demo/asr_tts/demo_asr/vosk-model-kz-0.15"  # Replace with your Vosk model path

    # Transcribe the WAV file
    transcription = transcribe_wav(wav_file_path, model_path)

    # Print the result
    print("Transcription:")
    print(transcription)

if __name__ == "__main__":
    main()