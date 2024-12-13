import io
import logging
import mimetypes
import os

# load packages
import tempfile
import time
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from transformers import AutoFeatureExtractor, WhisperModel

from hf_utils import load_custom_model_from_hf
from modules.audio import mel_spectrogram
from modules.bigvgan import bigvgan
from modules.campplus.DTDNN import CAMPPlus
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.postprocess import loudnorm, eq

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore")
VC_ROOT = "/data/ttd/seed-vc/"
os.environ["HF_HUB_CACHE"] = os.path.join(VC_ROOT, "./checkpoints/hf_cache")

# Load model and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16 = True


dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
    "Plachta/Seed-VC",
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
)
config = yaml.safe_load(open(dit_config_path, "r"))
model_params = recursive_munch(config["model_params"])
model = build_model(model_params, stage="DiT")
hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
sr = config["preprocess_params"]["sr"]

# Load checkpoints
model, _, _, _ = load_checkpoint(
    model,
    None,
    dit_checkpoint_path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
)
for key in model:
    model[key].eval()
    model[key].to(device)
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

# Load additional modules

campplus_ckpt_path = load_custom_model_from_hf(
    "funasr/campplus", "campplus_cn_common.bin", config_filename=None
)
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model.eval()
campplus_model.to(device)


bigvgan_model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False
)

# remove weight norm in the model and set to eval mode
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.eval().to(device)

# whisper

whisper_name = (
    model_params.speech_tokenizer.whisper_name
    if hasattr(model_params.speech_tokenizer, "whisper_name")
    else "openai/whisper-small"
)
whisper_model = WhisperModel.from_pretrained(
    whisper_name, torch_dtype=torch.float16
).to(device)
del whisper_model.decoder
whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

# Generate mel spectrograms
mel_fn_args = {
    "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
    "win_size": config["preprocess_params"]["spect_params"]["win_length"],
    "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
    "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": None,
    "center": False,
}


def to_mel(x):
    return mel_spectrogram(x, **mel_fn_args)


overlap_frame_len = 16

app = FastAPI()


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def get_svc_voice(actor, voice):
    return os.path.join(VC_ROOT, "models", actor, "train", voice + ".wav")


@torch.no_grad()
def voice_conversion(
    source,
    target,
    output_file,
    diffusion_steps,
    length_adjust,
    inference_cfg_rate,
    post_process=True,
):
    inference_module = model
    mel_fn = to_mel
    bigvgan_fn = bigvgan_model
    sr = 22050
    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]
    time_vc_start = time.time()

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        alt_inputs = whisper_feature_extractor(
            [converted_waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        alt_input_features = whisper_model._mask_input_features(
            alt_inputs.input_features, attention_mask=alt_inputs.attention_mask
        ).to(device)
        alt_outputs = whisper_model.encoder(
            alt_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        S_alt = alt_outputs.last_hidden_state.to(torch.float32)
        S_alt = S_alt[:, : converted_waves_16k.size(-1) // 320 + 1]
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[
                    :, traversed_time : traversed_time + 16000 * 30
                ]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        converted_waves_16k[
                            :,
                            traversed_time : traversed_time
                            + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            alt_inputs = whisper_feature_extractor(
                [chunk.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            alt_input_features = whisper_model._mask_input_features(
                alt_inputs.input_features, attention_mask=alt_inputs.attention_mask
            ).to(device)
            alt_outputs = whisper_model.encoder(
                alt_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            S_alt = alt_outputs.last_hidden_state.to(torch.float32)
            S_alt = S_alt[:, : chunk.size(-1) // 320 + 1]
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000
                if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    ori_inputs = whisper_feature_extractor(
        [ori_waves_16k.squeeze(0).cpu().numpy()],
        return_tensors="pt",
        return_attention_mask=True,
    )
    ori_input_features = whisper_model._mask_input_features(
        ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
    ).to(device)
    with torch.no_grad():
        ori_outputs = whisper_model.encoder(
            ori_input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    S_ori = ori_outputs.last_hidden_state.to(torch.float32)
    S_ori = S_ori[:, : ori_waves_16k.size(-1) // 320 + 1]

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = None
    shifted_f0_alt = None

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, codes, commitment_loss, codebook_loss = (
        inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
    )

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16 if fp16 else torch.float32
        ):
            # Voice Conversion
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = bigvgan_fn(vc_target.float())[0]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    time_vc_end = time.time()
    logger.info(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")
    if not post_process:
        torchaudio.save(output_file, vc_wave.cpu(), sr, format="wav")
        return True
    np_wave = vc_wave.cpu().numpy().flatten()
    np_wave = eq(np_wave, sr)
    np_wave, ori_loudness = loudnorm(np_wave, sr)
    logger.info(f"Original loudness: {ori_loudness:.2f} LUFS. normalized to -23 LUFS")
    sf.write(output_file, np_wave, sr, format="wav")
    return True


@app.post("/infer_vc")
async def infer_vc(
    actor: str,
    voice: str,
    steps: str = "50",
    post_process: bool = True,
    file: UploadFile = File(...),
):
    time_start = time.time()
    # 检查文件类型
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type is None or not mime_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only audio files are accepted."
        )

    args = {
        "diffusion_steps": int(steps),
        "length_adjust": 1.0,
        "inference_cfg_rate": 0.7,
    }

    wav_io = io.BytesIO()
    # 创建临时文件
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()  # 确保数据写入文件
            logger.info(
                "recieved vc request. actor: %s, size: %d", actor, len(contents)
            )
            # 调用 vc 方法
            voice_conversion(
                temp_file.name,
                get_svc_voice(actor, voice),
                wav_io,
                args["diffusion_steps"],
                args["length_adjust"],
                args["inference_cfg_rate"],
                post_process,
            )
            logger.info("vc done. cost time: %.03f", time.time() - time_start)

            wav_io.seek(0)
            # if post_process:
            #     wav_io = do_post_process(wav_io)
            #     logger.info(
            #         "post process done. cost time: %.03f", time.time() - time_start
            #     )

    except Exception as e:
        logger.exception("svc task failed: ")
        raise HTTPException(status_code=500, detail=str(e))
    if time.time() - time_start > 5:
        logger.warning(
            "svc task %s long time cost: %.03f", actor, time.time() - time_start
        )
    # 返回结果
    return StreamingResponse(
        wav_io,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7856)
