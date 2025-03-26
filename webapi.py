from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from funasr import AutoModel


model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    disable_update=True
)


# text = model.generate("")

# app = FastAPI()

# @app.post("/audio/transcriptions")
# async def transcribe_audio(
#     file: UploadFile = File(..., description="The audio file to transcribe"),
#     model: str = Form(..., description="The model to use for transcription"),
#     prompt: Optional[str] = Form(None, description="An optional text prompt to guide the transcription"),
#     response_format: Optional[str] = Form("json", description="The format of the transcription response (e.g., json, text)"),
#     temperature: Optional[float] = Form(0.0, description="Sampling temperature for the transcription"),
#     language: Optional[str] = Form(None, description="The language of the audio file (optional)")
# ):
#     """
#     Transcribe an audio file using the specified model and parameters.
#     """
#     # Placeholder for transcription logic
#     return {
#         "message": "Audio transcription endpoint is under development.",
#         "file_name": file.filename,
#         "model": model,
#         "prompt": prompt,
#         "response_format": response_format,
#         "temperature": temperature,
#         "language": language,
#     }


