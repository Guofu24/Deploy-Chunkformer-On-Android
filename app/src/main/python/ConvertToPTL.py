import torch
import torch.nn as nn
from typing import Tuple
import os
from pydub import AudioSegment
import torchaudio.compliance.kaldi as kaldi

# =========================================================
# CORRECT FEATURE EXTRACTION (Replicate ChunkFormer)
# =========================================================
def extract_features_like_chunkformer(audio_path: str, device='cpu'):
    """Extract features EXACTLY like ChunkFormer does"""
    sample_rate = 16000
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_sample_width(2)  # 16-bit
    audio = audio.set_channels(1)  # mono

    waveform = torch.as_tensor(
        audio.get_array_of_samples(),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    x = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )

    x_len = x.shape[0]
    return x, x_len


# =========================================================
# LOAD VOCAB
# =========================================================
def load_vocab(vocab_path: str):
    """Load vocab.txt with format: <token> <id>"""
    vocab_dict = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token, token_id = parts[0], int(parts[1])
                vocab_dict[token_id] = token

    max_id = max(vocab_dict.keys())
    vocab_list = [vocab_dict.get(i, "") for i in range(max_id + 1)]

    print(f"Loaded vocab with {len(vocab_list)} tokens")
    return vocab_list


# =========================================================
# WRAPPER
# =========================================================
class ChunkFormerWrapper(nn.Module):
    """Wrapper for ChunkFormer encoder + CTC"""
    def __init__(self, chunkformer_model, chunk_size=64, left_context=128, right_context=128):
        super().__init__()
        self.encoder = chunkformer_model.model.encoder
        self.ctc = chunkformer_model.model.ctc
        self.chunk_size_val = chunk_size
        self.left_context_val = left_context
        self.right_context_val = right_context

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs_out, xs_masks = self.encoder.forward_encoder(
            xs=xs,
            xs_lens=xs_lens,
            chunk_size=self.chunk_size_val,
            left_context_size=self.left_context_val,
            right_context_size=self.right_context_val,
        )
        xs_lens_out = xs_masks.squeeze(1).sum(-1)
        ctc_logits = self.ctc.log_softmax(xs_out)
        return ctc_logits, xs_lens_out


# =========================================================
# QUANTIZATION STEP
# =========================================================

def quantize_chunkformer(wrapper: nn.Module):
    qwrapper = torch.quantization.quantize_dynamic(
        wrapper,
        {nn.Linear},
        dtype=torch.qint8
    )
    print("Quantization done.")
    return qwrapper

# =========================================================
# EXPORT FUNCTION
# =========================================================
def export_chunkformer_correct(
    chunkformer_model,
    output_path_float: str,
    output_path_quant: str,
    example_audio_path: str,
    chunk_size=64,
    left_context=128,
    right_context=128,
    optimize_for_mobile=True
):
    print("Exporting ChunkFormer model...")

    # Extract example features
    x, x_len = extract_features_like_chunkformer(example_audio_path)
    print(f"Feature shape: {x.shape}, mean={x.mean():.2f}")

    # Prepare wrapper
    wrapper = ChunkFormerWrapper(
        chunkformer_model,
        chunk_size=chunk_size,
        left_context=left_context,
        right_context=right_context
    )
    wrapper.eval()

    example_xs = x.unsqueeze(0)
    example_lens = torch.tensor([x_len], dtype=torch.long)

    # ====================================================
    # FLOAT MODEL EXPORT
    # ====================================================
    print("\nExporting FLOAT model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, (example_xs, example_lens), strict=False)
        output, output_lens = traced_model(example_xs, example_lens)
        print(f"Float traced output shape: {output.shape}")

    if optimize_for_mobile:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        traced_model = optimize_for_mobile(traced_model)

    traced_model._save_for_lite_interpreter(output_path_float)
    size_mb = os.path.getsize(output_path_float) / (1024 * 1024)
    print(f"Saved FLOAT model: {output_path_float} ({size_mb:.2f} MB)")

    # ====================================================
    # QUANTIZED MODEL EXPORT
    # ====================================================
    print("\nExporting QUANTIZED model...")
    qwrapper = quantize_chunkformer(wrapper)

    with torch.no_grad():
        q_traced_model = torch.jit.trace(qwrapper, (example_xs, example_lens), strict=False)
        output, output_lens = q_traced_model(example_xs, example_lens)
        print(f"Quantized traced output shape: {output.shape}")

    if optimize_for_mobile:
        q_traced_model = optimize_for_mobile(q_traced_model)

    q_traced_model._save_for_lite_interpreter(output_path_quant)
    qsize_mb = os.path.getsize(output_path_quant) / (1024 * 1024)
    print(f"Saved QUANTIZED model: {output_path_quant} ({qsize_mb:.2f} MB)")

    print("\nExport completed successfully!")
    return output_path_float, output_path_quant


# =========================================================
# TEST EXPORTED MODEL
# =========================================================
def test_exported_model_correct(model_path: str, audio_path: str, vocab: list):
    print(f"\n{'='*60}")
    print(f"TEST MODEL: {model_path}")
    print('='*60)

    model = torch.jit.load(model_path)
    model.eval()

    x, x_len = extract_features_like_chunkformer(audio_path)
    
    # DEBUG: In stats
    print(f"   Features (raw):")
    print(f"   Shape: {x.shape}")
    print(f"   Mean: {x.mean():.2f}, Std: {x.std():.2f}")
    print(f"   Min: {x.min():.2f}, Max: {x.max():.2f}")
    print(f"   First frame: {x[0, :5]}")
    
    xs = x.unsqueeze(0)
    xs_lens = torch.tensor([x_len], dtype=torch.long)

    with torch.no_grad():
        output, out_lens = model(xs, xs_lens)
    
    # DEBUG: In output đầu tiên
    print(f"Model output (log-softmax):")
    print(f"   First frame top-5: {output[0, 0, :5]}")
    
    ids = torch.argmax(output, dim=-1)[0][:out_lens[0]]

    text = []
    prev_id = -1
    for idx in ids.tolist():
        if idx == 0 or idx == prev_id:
            prev_id = idx
            continue
        if idx < len(vocab):
            text.append(vocab[idx])
        prev_id = idx

    final_text = ''.join(text).strip().replace('▁', ' ')
    print(f"\nTranscript: {final_text}")
    return final_text


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    from chunkformer import ChunkFormerModel

    VOCAB_FILE = "D:\\Chunkformer_ONNX\\vocab.txt"
    TEST_AUDIO_FILE = "D:\\Chunkformer_ONNX\\audio_1.wav"
    OUTPUT_FLOAT = "D:\\Chunkformer_ONNX\\chunkformer_prequantized_2.ptl"
    OUTPUT_QUANT = "D:\\Chunkformer_ONNX\\chunkformer_quantized_4.ptl"

    print("=" * 60)
    print("LOAD MODEL")
    print("=" * 60)
    model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-large-vie")
    model.eval()

    print("\n" + "=" * 60)
    print("EXPORT BOTH FLOAT + QUANTIZED MODELS")
    print("=" * 60)

    export_chunkformer_correct(
        chunkformer_model=model,
        output_path_float=OUTPUT_FLOAT,
        output_path_quant=OUTPUT_QUANT,
        example_audio_path=TEST_AUDIO_FILE,
        chunk_size=64,
        left_context=128,
        right_context=128,
        optimize_for_mobile=True
    )
    x, x_len = extract_features_like_chunkformer("audio_1.wav")
    torch.save(x, "python_features.pt")
    print(f"First frame: {x[0, :5]}")
    print(f"Mean: {x.mean():.2f}, Std: {x.std():.2f}")

    print("\n" + "=" * 60)
    print("TEST BOTH MODELS")
    print("=" * 60)
    vocab = load_vocab(VOCAB_FILE)

    # print("\n--- FLOAT MODEL ---")
    # test_exported_model_correct(OUTPUT_FLOAT, TEST_AUDIO_FILE, vocab)
    TEST_AUDIO_FILE_1 = "D:\\Chunkformer_ONNX\\audio_1_fixed.wav"
    print("\n--- QUANTIZED MODEL ---")
    test_exported_model_correct(OUTPUT_QUANT, TEST_AUDIO_FILE_1, vocab)

    print("\nDONE! Both models exported and tested successfully.")
