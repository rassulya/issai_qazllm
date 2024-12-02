import queue
import sounddevice as sd
import sys
from vosk import Model, KaldiRecognizer
import time
import pandas as pd
import os


class AudioSpeechRecognition:
    def __init__(self, vosk_model_path, sample_rate, device=None):
        self.vosk_model = Model(vosk_model_path)
        self.sample_rate = sample_rate
        self.device = device
        self.q = queue.Queue()
        self.vosk_rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
        self.stream = None
        # self.stop_event = threading.Event()

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def start_stream(self):
        start_time = time.time()
        try:
            self.stream = sd.RawInputStream(samplerate=self.sample_rate,
                                            blocksize=8000,
                                            device=self.device,
                                            dtype="int16",
                                            channels=1,
                                            callback=self.callback)
            self.stream.start()
            silence_counter = 0

            while True:
                vosk_data = self.q.get()
                if self.vosk_rec.AcceptWaveform(vosk_data):
                    vosk_output = self.vosk_rec.Result().split('"')[-2]
                    if vosk_output:
                        print("User (kz):", vosk_output)
                        duration = time.time() - start_time
                        self.log_time('ASR_new', duration)
                        return vosk_output
                else:
                    partial_result = self.vosk_rec.PartialResult().split('"')[-2]
                    # if not partial_result:
                    #     # silence_counter += 1
                    #     # if silence_counter > 500:  # Adjust the threshold as needed
                    #     #     print("No speech detected. Quitting...")
                    #     #     return None, None
                    # else:
                    #     # silence_counter = 0
                    #     print("Partial result:", partial_result)
                    #     start_time = time.time()
                    if partial_result:
                        print("Partial result:", partial_result)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_recognizer()
            self.stop_stream()

    def reset_recognizer(self):
        """Reset the recognizer state to avoid carrying over any previous partial results."""
        self.vosk_rec = KaldiRecognizer(self.vosk_model, self.sample_rate)

    def stop_stream(self):
        """Ensure resources are released."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Audio stream stopped and resources released.")

    def reset_flag(self):
        self.flag = True

    def log_time(self, method: str, duration: float):
        filename = f'{method}_times.csv'
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=[method])
        
        new_row = {method: duration}
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        df.to_csv(filename, index=False)

# Example usage
if __name__ == "__main__":
    vosk_model_path = "vosk/vosk-model-kz-0.15"
    sample_rate = 16000
    asr = AudioSpeechRecognition(vosk_model_path, sample_rate)
    vosk_output = asr.start_stream()
    if vosk_output:
        print("Detected speech:", vosk_output)
    else:
        print("No speech detected.")
