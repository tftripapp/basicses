import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from vosk import Model, KaldiRecognizer
import wave
import subprocess
import tempfile

MODEL_PATH = "model-tr"

app = Flask(__name__)
CORS(app)

# Model klasörü kontrolü
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("model-tr klasörü bulunamadı! Lütfen Türkçe Vosk modelini indirip bu klasöre koyun.")

model = Model(MODEL_PATH)

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
    wf.close()
    os.remove(temp_path)
    os.remove(wav_path)
    return jsonify({'text': text.strip()})

@app.route("/")
def index():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 
