from IPython import embed
import logging

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
        self.model = Wav2Vec2ForCTC.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]

        # input_values = self.processor(wavs, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(device)
        input_values = self.processor(
            wavs,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            sampling_rate=SAMPLE_RATE,
        ).input_values.to(device)

        output_values = self.model(input_values, output_hidden_states=True)
        return {"hidden_states": output_values.hidden_states}
