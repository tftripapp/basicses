import os
import urllib.request
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from vosk import Model, KaldiRecognizer
import wave
import subprocess
import tempfile

MODEL_PATH = "model-tr"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.4.zip"

app = Flask(__name__)
CORS(app)

# Modeli indir ve çıkar
if not os.path.exists(MODEL_PATH):
    import zipfile
    print("Türkçe modeli indiriliyor...")
    zip_path = "model-tr.zip"
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    # Model klasörü ismini düzelt
    for d in os.listdir('.'):
        if d.startswith('vosk-model-small-tr'):
            os.rename(d, MODEL_PATH)
    os.remove(zip_path)

model = Model(MODEL_PATH)

# Ses dosyasını wav'a çevir

def convert_to_wav(input_path):
    output_path = tempfile.mktemp(suffix='.wav')
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000', '-ac', '1', '-f', 'wav', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        temp_path = temp.name
    # wav'a çevir
    wav_path = convert_to_wav(temp_path)
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        return jsonify({'error': 'Ses dosyası uygun formatta değil'}), 400
    rec = KaldiRecognizer(model, wf.getframerate())
    result = ''
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part = rec.Result()
            result += part
    result += rec.FinalResult()
    import json as js
    try:
        text = ''
        for line in result.split('}{'):
            if not line.startswith('{'):
                line = '{' + line
            if not line.endswith('}'):
                line = line + '}'
            res = js.loads(line)
            if 'text' in res:
                text += res['text'] + ' '
    except Exception:
        text = result
    # Temizlik
    wf.close()
    os.remove(temp_path)
    os.remove(wav_path)
    return jsonify({'text': text.strip()})

@app.route("/")
def index():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 
