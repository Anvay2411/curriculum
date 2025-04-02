import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr

model_path = "hindi.nemo"
lang_id = "hi"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval() # inference mode
model = model.to(device) # transfer model to device


model.cur_decoder = "ctc"
ctc_text = model.transcribe(['./samples/modi_speech.wav'], batch_size=1,logprobs=False, language_id=lang_id)[0]
print(ctc_text)