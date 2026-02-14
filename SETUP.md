# Windows virtual environment setup

## Prerequisites

- **Python 3.9 or 3.10** (3.11 may work; TensorFlow 2.15 is tested on 3.9–3.11).
- **pip** (usually with Python).

## Steps

### 1. Open a terminal in the project root

```powershell
cd "d:\AiInterview Platform\AiInterview-Platform"
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If you see an execution policy error, run once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run the activation again.

You should see `(venv)` in the prompt.

### 4. Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

**Note:** PyAudio on Windows sometimes needs a prebuilt wheel. If `pip install -r requirements.txt` fails on PyAudio:

- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if needed, or
- Use a wheel from [Christoph Gohlke’s page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) (e.g. `pip install PyAudio‑0.2.14‑cp310‑cp310‑win_amd64.whl` for Python 3.10 64‑bit).

### 6. Run the Streamlit app

```powershell
streamlit run app.py
```

The app will open in your browser (e.g. http://localhost:8501).

### 7. Deactivate when done

```powershell
deactivate
```

---

## Folder structure

```
AiInterview-Platform/
├── app.py                 # Streamlit entry point
├── requirements.txt
├── SETUP.md               # This file
├── analyzer/
│   ├── __init__.py
│   ├── face_detection.py  # OpenCV + MediaPipe
│   ├── emotion_detection.py  # Keras .h5
│   ├── voice_analysis.py  # Librosa + SpeechRecognition
│   └── scoring.py         # Combined confidence
└── models/
    └── emotion_model.h5   # Place your pretrained model here
```

## Model

Put your pretrained emotion detection model at `models/emotion_model.h5`. The `analyzer.emotion_detection` module expects a Keras model that accepts the input shape you used during training (e.g. `(48, 48, 1)` for grayscale faces).
