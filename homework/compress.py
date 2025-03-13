from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
#         raise NotImplementedError()
        with torch.no_grad():
            tokens = self.tokenizer.encode(x.unsqueeze(0))
        tokens = tokens.squeeze(0)
        seq_len = tokens.shape[0]
        vocab_size = self.autoregressive.vocab_size

        cdfs = []
        for i in range(seq_len):
            if i == 0:
                prev_tokens = torch.tensor([], dtype=torch.long, device=x.device)
            else:
                prev_tokens = tokens[:i]
            with torch.no_grad():
                logits = self.autoregressive(prev_tokens.unsqueeze(0))
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            cdf = torch.cat([torch.zeros(1, device=probs.device), torch.cumsum(probs, dim=0)])
            cdfs.append(cdf.unsqueeze(0))

        cdfs_tensor = torch.cat(cdfs, dim=0).cpu()
        symbols = tokens.cpu().to(torch.int16)

        byte_stream = torchac.encode_float_cdf(cdfs_tensor, symbols, check_input_bounds=True)
        return byte_stream

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
#         raise NotImplementedError()
        h, w = 150, 100
        patch_size = self.tokenizer.patch_size
        seq_len = (h // patch_size) * (w // patch_size)

        symbols = []
        device = self.autoregressive.device
        for i in range(seq_len):
            prev_tokens = torch.tensor(symbols, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.autoregressive(prev_tokens.unsqueeze(0))  # [1, vocab_size]
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            cdf = torch.cat([torch.zeros(1, device=device), torch.cumsum(probs, dim=0)]).cpu().unsqueeze(0)
            # Decode one symbol at a time
            symbol = torchac.decode_float_cdf(cdf, x[i:(i+1)] if i == 0 else x)  # Simplified example
            symbols.append(symbol.item())

        tokens = torch.tensor(symbols, dtype=torch.long, device=device)
        with torch.no_grad():
            reconstructed = self.tokenizer.decode(tokens.unsqueeze(0))
        return reconstructed.squeeze(0)

def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
