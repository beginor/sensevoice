from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from funasr import AutoModel
import numpy as np
import torch
import torchaudio


device = "cuda:0"

auto_model = AutoModel(
    model="iic/SenseVoiceSmall",
    device=device,
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    disable_update=True,
)

app = FastAPI()


emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()


def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()


@app.post(
    "/v1/audio/transcriptions",
    tags=['Audio'],
)
async def transcribe_audio(
    file: UploadFile = File(),
    model: str | None = Form('SenseVoiceSmall'),
    prompt: str | None = Form(''),
    temperature: float | None = Form(0.0),
    language: str | None = Form('zh'),
):
    print(f'filename: {file.filename}')
    print(f'content_type: {file.content_type}')
    print(f'model: {model}')
    print(f'prompt: {prompt}')
    print(f'temperature: {temperature}')
    print(f'language: {language}')

    buffer = await file.read()
    # 加载音频
    tensor, sample_rate = torchaudio.load(
        BytesIO(buffer),
        backend= 'ffmpeg' if file.filename.endswith('.mp3') else None,
    )
    print(f'tensor: {tensor}')
    # 转换为 NumPy 数组并归一化
    input_wav = tensor.numpy()
    if input_wav.ndim > 1:
        input_wav = input_wav.mean(axis=0)

    input_wav = input_wav / np.max(np.abs(input_wav))
    # print(f'input_wav: {input_wav}')

    print(f'sample_rate: {sample_rate}')
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
        input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

    merge_vad = True
    print(f"language: {language}, merge_vad: {merge_vad}")
    parts = auto_model.generate(
        input=input_wav,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=merge_vad,
    )

    result = parts[0]
    print(f'result: {result}')

    text = format_str_v3(result["text"])

    print(f'text: {text}')

    return { 'text': text }
