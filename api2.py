import io
import logging
import mimetypes
import os

# load packages
import tempfile
import time
import warnings
import shutil

from fastapi.params import Form
import soundfile as sf
import uvicorn
from fastapi import File, HTTPException, UploadFile, FastAPI
from fastapi.responses import StreamingResponse

from modules.postprocess import loudnorm, eq

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
warnings.simplefilter("ignore")
VC_ROOT = "/data/ttd/seed-vc/"
os.environ["HF_HUB_CACHE"] = os.path.join(VC_ROOT, "./checkpoints/hf_cache")

from app import voice_conversion

def post_process_file(vc_wave, sr: int):
    output_file = io.BytesIO()
    try:
        target_loudness = -23.0  # 设置目标 LUFS 声度
        np_wave = eq(vc_wave, sr)
        np_wave, ori_loudness = loudnorm(np_wave, sr, target_loudness)
        logger.info(f"Original loudness: {ori_loudness:.2f} LUFS. normalized to {target_loudness:.2f} LUFS")
    except Exception as e:
        logger.warning(f"Post-processing failed with error: {e}")
    sf.write(output_file, np_wave, sr, format="wav")
    output_file.seek(0)
    return output_file

def get_svc_voice(actor, voice):
    vc_ref_file = os.path.join(VC_ROOT, "models", actor, "train", voice + ".wav")
    if not os.path.exists(vc_ref_file):
        raise FileNotFoundError(f"Voice file not found: {vc_ref_file}")
    return vc_ref_file

app = FastAPI()

@app.post("/infer_vc")
async def infer_vc(
    actor: str,
    voice: str,
    steps: str = "50",
    post_process: bool = True,
    file: UploadFile = File(...)
):
    time_start = time.time()
    # 检查文件类型
    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type is None or not mime_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only audio files are accepted."
        )

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
            vc_generator = voice_conversion(
                temp_file.name,
                get_svc_voice(actor, voice),
                int(steps),
                1.0,
                0.7,
                False,
                False,
                0,
            )
            # 获取生成器的最终结果
            final_result = next((r for _, r in vc_generator if r is not None), None)
            if not final_result: raise HTTPException(status_code=500, detail="处理音频失败")
            sr, np_wav = final_result

            logger.info("vc done. cost time: %.03f", time.time() - time_start)

    except FileNotFoundError as e:
        logger.exception("svc task failed: ")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("svc task failed: ")
        raise HTTPException(status_code=500, detail=str(e))
    if time.time() - time_start > 5:
        logger.warning(
            "svc task %s long time cost: %.03f", actor, time.time() - time_start
        )
    # 返回结果
    return StreamingResponse(
        post_process_file(np_wav, sr),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )

@app.post("/svc_file")
async def svc_file(
    src_file: UploadFile = File(...),
    ref_file: UploadFile = File(...),
    steps: str = "50",
    length_adjust: str = "1.0",
    inference_cfg_rate: str = "0.7",
    f0_conditioned: bool = False,
    auto_f0_adjust: bool = False,
    pitch_shift: str = "0",
):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file_src:
        contents = await src_file.read()
        temp_file_src.write(contents)
        temp_file_src.flush()  # 确保数据写入文件

    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_file_ref:
        contents = await ref_file.read()
        temp_file_ref.write(contents)
        temp_file_ref.flush()  # 确保数据写入文件

    try: 
        # 处理上传的文件
        vc_generator = voice_conversion(
            temp_file_src.name, 
            temp_file_ref.name,
            diffusion_steps=int(steps),
            length_adjust=float(length_adjust),
            inference_cfg_rate=float(inference_cfg_rate),
            f0_condition=f0_conditioned,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=int(pitch_shift),
        )
        
        # 获取生成器的最终结果
        final_result = next((r for _, r in vc_generator if r is not None), None)
        if not final_result: raise HTTPException(status_code=500, detail="处理音频失败")
        sr, np_wav = final_result
        
        return StreamingResponse(
            post_process_file(np_wav, sr),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )
        
    except Exception as e:
        logger.exception("svc task failed: ")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7856)
