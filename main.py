from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf
import streamlit as st

# ------------------- Модель -------------------
class PlaceAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(PlaceAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()
        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, mel, time)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x

# ------------------- Настройки -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=64,
    f_min=0,
    f_max=8000,
    power=2.0
)

# Загрузка классов и модели
labels = torch.load('region_labels.pth')
model = PlaceAudio(num_classes=len(labels))
model.load_state_dict(torch.load('region_model.pth', map_location=device))
model.to(device)
model.eval()

max_len = 500

# ------------------- Обработка аудио -------------------
def change_audio_format(waveform, sample_rate):
    # Приводим тип к float32 тензору
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Если стерео — усредняем до моно
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Приводим sample rate
    if sample_rate != 16000:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    # Создаём Mel-спектрограмму
    spec = transform(waveform).squeeze(0)

    # Подгоняем длину
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))

    return spec

# ------------------- FastAPI -------------------
check_audio = FastAPI(title='Place Audio Classifier')


name = st.radio("Choose input method:", ["Upload", "Record"], horizontal=True)
if name == 'Upload':
    st.title('Model Region')
    st.text('Загрузите аудио файл')

    audio_file = st.file_uploader('Выбериту файл', type='wav')

    if not audio_file:
        st.warning('Загрузите .wav файл')
    else:
        st.audio(audio_file)
    if st.button('Распознать'):
        try:
            data = audio_file.read()
            if not data:
                raise HTTPException(status_code=400, detail='Data not found')

            wf, sr = sf.read(io.BytesIO(data), dtype='float32')
            spec = change_audio_format(wf, sr).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(spec)
                pred_idx = torch.argmax(y_pred, dim=1).item()
                pred_class = labels[pred_idx]

            st.success({'Class': pred_class})

        except Exception as e:
            st.exception(e)

elif name == 'Record':
    st.title('Model Region')
    st.info(f'Скажи слово из этого списка: {labels}')
    audio_record = st.audio_input('Скажите слово')
    if st.button('Распознать'):
        try:
            data = audio_record.read()
            if not data:
                raise HTTPException(status_code=400, detail='Data not found')

            wf, sr = sf.read(io.BytesIO(data), dtype='float32')
            spec = change_audio_format(wf, sr).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(spec)
                pred_idx = torch.argmax(y_pred, dim=1).item()
                pred_class = labels[pred_idx]

            st.success({'Class': pred_class})

        except Exception as e:
            st.exception(e)





