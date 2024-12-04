import pandas as pd
import os
from functools import partial
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union
import numpy as np
import onnxruntime
from espeak_phonemizer import Phonemizer
import logging
from functools import partial

fs = 22050

_FILE = Path(__file__)
_DIR = _FILE.parent
_LOGGER = logging.getLogger(_FILE.stem)

_BOS = "^"
_EOS = "$"
_PAD = "_"

logger = logging.getLogger(__name__)

@dataclass
class PiperConfig:
    num_symbols: int
    num_speakers: int
    sample_rate: int
    espeak_voice: str
    length_scale: float
    noise_scale: float
    noise_w: float
    phoneme_id_map: Mapping[str, Sequence[int]]

class Piper:
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        if config_path is None:
            config_path = f"{model_path}.json"

        if not Path(config_path).exists():
            print(f"Warning: Config file {config_path} not found. Using default configuration.")
            self.config = self.default_config()
        else:
            self.config = load_config(config_path)

        self.phonemizer = Phonemizer(self.config.espeak_voice)
        self.model = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=onnxruntime.SessionOptions(),
            providers=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
            ],
        )

    @staticmethod
    def default_config():
        return PiperConfig(
            num_symbols=256,
            num_speakers=1,
            sample_rate=22050,
            espeak_voice="kk",  # Kazakh voice
            length_scale=1.0,
            noise_scale=0.667,
            noise_w=0.8,
            phoneme_id_map={},
        )

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        if length_scale is None:
            length_scale = self.config.length_scale
        if noise_scale is None:
            noise_scale = self.config.noise_scale
        if noise_w is None:
            noise_w = self.config.noise_w

        phonemes_str = self.phonemizer.phonemize(text)
        phonemes = [_BOS] + list(phonemes_str) + [_EOS]
        phoneme_ids: List[int] = []

        if not self.config.phoneme_id_map:
            phoneme_ids = [ord(p) for p in phonemes]
        else:
            for phoneme in phonemes:
                phoneme_ids.extend(self.config.phoneme_id_map.get(phoneme, [ord(phoneme)]))
                phoneme_ids.extend(self.config.phoneme_id_map.get(_PAD, [ord(_PAD)]))

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        input_feed = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        # Проверяем, есть ли 'sid' в списке входных параметров модели
        if speaker_id is not None and 'sid' in [input.name for input in self.model.get_inputs()]:
            input_feed["sid"] = np.array([speaker_id], dtype=np.int64)

        audio = self.model.run(
            None,
            input_feed,
        )[0].squeeze((0, 1))
        return audio, self.config.sample_rate

def load_config(config_path: Union[str, Path]) -> PiperConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
        inference = config_dict.get("inference", {})

        return PiperConfig(
            num_symbols=config_dict["num_symbols"],
            num_speakers=config_dict["num_speakers"],
            sample_rate=config_dict["audio"]["sample_rate"],
            espeak_voice=config_dict["espeak"]["voice"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            phoneme_id_map=config_dict["phoneme_id_map"],
        )


# def audio_float_to_int16(
#     audio: np.ndarray, max_wav_value: float = 32767.0
# ) -> np.ndarray:
#     """Normalize audio and convert to int16 range"""
#     audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
#     audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
#     audio_norm = audio_norm.astype("int16")
#     return audio_norm


class TTS:
    def __init__(self, model_path, voice_id) -> None:
        self.model_path = model_path
        self.voice_id = voice_id
        self.voice = Piper(self.model_path)
        
        logger.debug(f"Initializing TTS with model_path: {model_path}, voice_id: {voice_id}")
        
        # Create different versions of the synthesize method based on the presence of voice_id
        if self.voice_id is not None:
            logger.debug(f"Creating synthesize method with speaker_id: {self.voice_id}")
            self.synthesize = partial(
                self.voice.synthesize,
                speaker_id=self.voice_id,
                length_scale=None,
                noise_scale=None,
                noise_w=None,
            )
        else:
            logger.debug("Creating synthesize method without speaker_id")
            self.synthesize = partial(
                self.voice.synthesize,
                length_scale=None,
                noise_scale=None,
                noise_w=None,
            )

    def generate_audio(self, text):
        logger.debug(f"Generating audio for text: {text[:50]}...")
        audio_norm, sample_rate = self.synthesize(text)
        return audio_norm, sample_rate
    
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
