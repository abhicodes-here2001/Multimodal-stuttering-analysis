# 🎙️ Multimodel Stuttering Analysis System# 🎙️ Multimodal Stuttering Analysis System



A comprehensive AI-powered clinical tool for analyzing speech disfluencies (stuttering) using multimodal deep learning. Combines **WavLM** for acoustic stutter detection and **Whisper** for transcription to generate detailed clinical reports.> **AI-powered speech analysis tool for detecting and classifying stuttering patterns using deep learning**



![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

![React](https://img.shields.io/badge/React-18+-61DAFB.svg)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Features

## 🌟 Motivation

### Stutter Detection (5 Types)

| Type | Description | Example |**68 million people worldwide** struggle with stuttering, yet access to speech therapy remains limited and expensive. Traditional assessment methods are:

|------|-------------|---------|- **Subjective** - Relying on clinician interpretation

| **Prolongation** | Extended sound duration | "Ssssssnake" |- **Time-consuming** - Manual analysis of speech patterns

| **Block** | Complete speech stoppage with tension | Silent pause mid-word |- **Inaccessible** - Requiring in-person specialist visits

| **Sound Repetition** | Repeating sounds/syllables | "B-b-b-ball" |

| **Word Repetition** | Repeating whole words | "I-I-I want" |This project was born from a simple question: *Can AI democratize stuttering assessment and make it accessible to everyone?*

| **Interjection** | Filler words | "um", "uh", "like" |

By combining state-of-the-art speech recognition with specialized stutter detection models, this system provides **objective, instant, and accessible** stuttering analysis that can assist:

### Clinical Report Includes- Speech-language pathologists in clinical settings

- ✅ **Severity Assessment** (1-5 scale based on % Words Stuttered)- Researchers studying speech disorders

- ✅ **Word-level stutter mapping** with timestamps- Individuals seeking self-assessment tools

- ✅ **Stutter type distribution** (pie chart)

- ✅ **Temporal pattern analysis** (timeline view)---

- ✅ **Probability heatmap** for each stutter type

- ✅ **Clinical recommendations** based on patterns## 🎯 What This Project Does

- ✅ **Downloadable JSON report**

This is a **multimodal AI system** that analyzes audio recordings to:

## 🏗️ Architecture

1. **Detect Stuttering Events** - Identifies where stuttering occurs in speech

```2. **Classify Stutter Types** - Categorizes into 5 clinical stuttering types

┌─────────────────┐     ┌─────────────────┐3. **Transcribe Speech** - Provides word-level transcription with timestamps

│   Audio Input   │────▶│  Preprocessing  │4. **Generate Clinical Reports** - Combines analysis into actionable insights

└─────────────────┘     │  (16kHz mono)   │

                        └────────┬────────┘### Stuttering Types Detected

                                 │

                    ┌────────────┴────────────┐| Type | Description | Example |

                    ▼                         ▼|------|-------------|---------|

           ┌───────────────┐         ┌───────────────┐| **Prolongation** | Stretching sounds | "Sssssnake" |

           │    WavLM      │         │    Whisper    │| **Block** | Silent pauses, inability to produce sound | "I want to... ... ...go" |

           │ (Stutter Det) │         │ (Transcribe)  │| **Sound Repetition** | Repeating individual sounds | "B-b-b-ball" |

           └───────┬───────┘         └───────┬───────┘| **Word Repetition** | Repeating whole words | "I want want want to go" |

                   │                         │| **Interjection** | Filler words/sounds | "Um, uh, like, you know" |

                   └──────────┬──────────────┘

                              ▼---

                    ┌─────────────────┐

                    │  Clinical Report │## 🏗️ System Architecture

                    │   Generator      │

                    └─────────────────┘```

```┌─────────────────────────────────────────────────────────────────┐

│                        USER INPUT                                │

## 📁 Project Structure│                    (Audio File Upload)                           │

└─────────────────────────────────────────────────────────────────┘

```                              │

Multimodal-stuttering-analysis/                              ▼

├── models/┌─────────────────────────────────────────────────────────────────┐

│   └── WaveLm_model.py      # WavLM classification model│                     PREPROCESSING                                │

├── training/│  ┌─────────────────┐    ┌─────────────────┐                     │

│   ├── Dataset.py           # SEP-28k dataset loader│  │  Audio Check    │ →  │   Chunking      │                     │

│   └── train_waveLM.py      # Training script│  │  (16kHz, Mono)  │    │   (3-sec clips) │                     │

├── inference/│  └─────────────────┘    └─────────────────┘                     │

│   ├── WaveLM_inference.py  # Stutter detection└─────────────────────────────────────────────────────────────────┘

│   ├── Whisper_inference.py # Transcription                              │

│   └── clinical_report.py   # Report generation              ┌───────────────┴───────────────┐

├── Preprocessing/              ▼                               ▼

│   ├── audio_check.py       # Audio validation┌──────────────────────────┐    ┌──────────────────────────┐

│   └── chunking.py          # Audio segmentation│      WavLM Model         │    │     Whisper Model        │

├── backend/│   (Stutter Detection)    │    │    (Transcription)       │

│   └── api.py               # FastAPI REST API│                          │    │                          │

├── frontend/│  • 94M total parameters  │    │  • OpenAI Whisper Base   │

│   └── src/                 # React UI│  • 7.1M trainable        │    │  • Word-level timestamps │

├── checkpoints/             # Trained model weights│  • 12 Transformer layers │    │  • Multi-language        │

├── data/│  • 768-dim hidden states │    │                          │

│   ├── SEP-28k_labels.csv   # Dataset labels└──────────────────────────┘    └──────────────────────────┘

│   └── clips/               # Audio samples              │                               │

└── requirements.txt              └───────────────┬───────────────┘

```                              ▼

┌─────────────────────────────────────────────────────────────────┐

## 🚀 Quick Start│                    CLINICAL REPORT                               │

│  ┌─────────────────────────────────────────────────────────┐    │

### Prerequisites│  │  • Stutter type classification with confidence scores   │    │

- Python 3.9+│  │  • Word-by-word analysis with timestamps                │    │

- Node.js 18+│  │  • Severity assessment                                  │    │

- ~4GB disk space for models│  │  • Recommendations                                      │    │

│  └─────────────────────────────────────────────────────────┘    │

### Installation└─────────────────────────────────────────────────────────────────┘

```

```bash

# Clone repository---

git clone https://github.com/abhicodes-here2001/Multimodal-stuttering-analysis.git

cd Multimodal-stuttering-analysis## 🧠 Model Details



# Create virtual environment### WavLM - Stutter Classification Model

python -m venv env

source env/bin/activate  # On Windows: env\Scripts\activate| Component | Details |

|-----------|---------|

# Install Python dependencies| **Base Model** | `microsoft/wavlm-base` |

pip install -r requirements.txt| **Architecture** | Transformer Encoder |

| **Total Parameters** | 94,404,613 (~94M) |

# Install frontend dependencies| **Trainable Parameters** | 7,091,717 (~7.1M) |

cd frontend| **Frozen Parameters** | 87,312,896 (~87M) |

npm install| **Training Strategy** | Transfer Learning with partial unfreezing |

cd ..| **Unfrozen Layers** | Last encoder layer + Classification head |

```| **Classification Head** | Linear(768 → 5) with Sigmoid |

| **Loss Function** | BCEWithLogitsLoss (multi-label) |

### Running the Application| **Optimizer** | AdamW |



**Terminal 1 - Backend:**### Whisper - Speech Transcription

```bash

source env/bin/activate| Component | Details |

cd backend|-----------|---------|

uvicorn api:app --host 0.0.0.0 --port 8000 --reload| **Model** | OpenAI Whisper Base |

```| **Parameters** | ~74M |

| **Capability** | Speech-to-text with word timestamps |

**Terminal 2 - Frontend:**| **Languages** | 99+ languages supported |

```bash| **Usage** | Inference only (pre-trained) |

cd frontend

npm start---

```

## 📊 Training Details

Open http://localhost:3000 in your browser.

### Dataset

## 📊 Model Training- **Name**: SEP-28k (Stuttering Events in Podcasts)

- **Total Samples**: ~28,000 labeled audio clips

The WavLM model was fine-tuned on the SEP-28k dataset:- **Valid Samples Used**: ~20,866 (after filtering missing files)

- **Split**: 80% Training / 20% Validation

```bash- **Sources**: YouTube podcasts featuring people who stutter

python training/train_waveLM.py

```### Training Configuration



**Training Results:**```python

- Validation Accuracy: **79.55%**BATCH_SIZE = 16

- Best model saved to: `checkpoints/wavlm_stutter_classification_best.pth`LEARNING_RATE = 5e-5

EPOCHS = 15

## 🔧 API EndpointsDEVICE = "mps"  # Apple Silicon GPU

```

| Endpoint | Method | Description |

|----------|--------|-------------|### Training Strategy

| `/analyze` | POST | Upload audio, get clinical report |

| `/health` | GET | Check server status |1. **Freeze Base Model** - Preserve pre-trained speech representations

| `/report/{id}` | GET | Retrieve saved report |2. **Unfreeze Last Layer** - Allow fine-tuning of task-specific features

3. **Train Classification Head** - Learn stutter type mapping

### Example API Usage

### 📈 Training Results

```python

import requests#### Performance Summary



files = {'file': open('audio.wav', 'rb')}| Metric | Value |

data = {|--------|-------|

    'patient_name': 'John Doe',| **Best Validation Accuracy** | **79.55%** |

    'patient_id': 'PT-001',| **Best Validation Loss** | **0.4446** |

    'threshold': 0.4| **Best Epoch** | 12 |

}| **Total Epochs** | 15 |

| **Training Samples** | 16,693 |

response = requests.post('http://localhost:8000/analyze', files=files, data=data)| **Validation Samples** | 4,173 |

report = response.json()

print(f"Severity: {report['severity']['label']}")#### Epoch-by-Epoch Progress

```

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Model Saved |

## 📈 Severity Classification|-------|------------|-----------|----------|---------|-------------|

| 1 | 0.5429 | 73.16% | 0.5018 | 77.10% | ✅ |

Based on % Words Stuttered (%WS):| 2 | 0.4785 | 77.55% | 0.4764 | 77.84% | ✅ |

| 3 | 0.4619 | 78.47% | 0.4607 | 78.32% | ✅ |

| Score | Label | Criteria || 4 | 0.4517 | 79.01% | 0.4571 | 78.56% | ✅ |

|-------|-------|----------|| 5 | 0.4454 | 79.35% | 0.4575 | 78.41% | |

| 1 | Very Mild | < 5% words stuttered || 6 | 0.4378 | 79.70% | 0.4635 | 78.29% | |

| 2 | Mild | 5-10% words stuttered || 7 | 0.4310 | 80.00% | 0.4562 | 78.41% | |

| 3 | Moderate | 10-20% words stuttered || 8 | 0.4270 | 80.21% | 0.4520 | 78.87% | ✅ |

| 4 | Severe | 20-30% words stuttered || 9 | 0.4229 | 80.38% | 0.4494 | 79.07% | |

| 5 | Very Severe | > 30% words stuttered || 10 | 0.4194 | 80.52% | 0.4473 | 79.31% | |

| 11 | 0.4150 | 80.72% | 0.4463 | 79.36% | ✅ |

## 🌐 Deployment| 12 | 0.4119 | 80.87% | **0.4446** | **79.55%** | ✅ |

| 13 | 0.4097 | 80.93% | 0.4472 | 79.41% | |

### Hugging Face Spaces (Gradio)| 14 | 0.4061 | 81.08% | 0.4457 | 79.48% | |

| 15 | 0.4043 | 81.18% | 0.4478 | 79.29% | |

See `huggingface/` folder for deployment files:

- `app.py` - Gradio interface#### Training Analysis

- `requirements.txt` - Dependencies

```

### Docker (Coming Soon)Training Curve Visualization:



```bashAccuracy Progress:

docker build -t stutter-analysis .Epoch 1  ████████████████████████████████░░░░░░░░ 77.10%

docker run -p 8000:8000 stutter-analysisEpoch 5  █████████████████████████████████░░░░░░░ 78.41%

```Epoch 10 ██████████████████████████████████░░░░░░ 79.31%

Epoch 12 ███████████████████████████████████░░░░░ 79.55% ← Best

## 🤝 ContributingEpoch 15 ██████████████████████████████████░░░░░░ 79.29%

```

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/amazing-feature`)#### Key Observations

3. Commit changes (`git commit -m 'Add amazing feature'`)

4. Push to branch (`git push origin feature/amazing-feature`)1. **Healthy Learning Curve** 📊

5. Open a Pull Request   - Training accuracy improved steadily: 73.16% → 81.18%

   - Validation accuracy improved consistently: 77.10% → 79.55%

## 📜 License   - No sudden drops or instabilities



This project is licensed under the MIT License - see [LICENSE](LICENSE) file.2. **No Significant Overfitting** ✅

   - Train-Val accuracy gap at best epoch: ~1.3% (80.87% - 79.55%)

## 🙏 Acknowledgments   - Final gap: ~1.9% (81.18% - 79.29%)

   - Gap remains small throughout training

- [SEP-28k Dataset](https://github.com/apple/ml-stuttering-events-dataset) - Apple

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - Microsoft3. **Optimal Checkpoint Selection** 🎯

- [Whisper](https://github.com/openai/whisper) - OpenAI   - Best model saved at epoch 12 (not final epoch)

   - Early stopping behavior: validation improved for 12 epochs

## 📧 Contact   - Model saved 7 times during training (epochs 1,2,3,4,8,11,12)



**Abhisar Gautam** - [@abhicodes-here2001](https://github.com/abhicodes-here2001)4. **Convergence Analysis** 📉

   - Loss converged smoothly without oscillations

---   - Learning rate (5e-5) proved appropriate for fine-tuning

   - Partial unfreezing strategy effective

⭐ Star this repo if you find it useful!

#### Hardware & Training Time

| Specification | Value |
|---------------|-------|
| **Device** | Apple M1 Pro (MPS) |
| **Memory** | 16GB Unified Memory |
| **Training Time** | ~45 minutes |
| **Time per Epoch** | ~3 minutes |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- macOS (Apple Silicon) / Linux / Windows
- ~4GB disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/abhicodes-here2001/Multimodal-stuttering-analysis.git
cd Multimodal-stuttering-analysis

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Checkpoints (Git LFS)

The trained model weights (`checkpoints/*.pth`) are stored using **Git LFS** (Large File Storage) because they are ~360MB each.

```bash
# If you haven't installed Git LFS yet:
brew install git-lfs  # macOS
# or: apt-get install git-lfs  # Ubuntu

# Initialize Git LFS
git lfs install

# Pull the model files (they will download automatically on clone)
git lfs pull
```

> **Note**: The model checkpoint `wavlm_stutter_classification_best.pth` is required for inference. Without it, the system will use random weights and produce inaccurate results.

### Quick Start

```python
from inference.whisper_inference import transcribe_audio, load_whisper_model
from inference.wavlm_inference import predict_stutter, load_wavlm_model

# Load models
whisper_model = load_whisper_model("base")
wavlm_model = load_wavlm_model("checkpoints/wavlm_stutter_classification_best.pth")

# Analyze audio
transcription = transcribe_audio("your_audio.wav", model=whisper_model)
stutter_results = predict_stutter("your_audio.wav", model=wavlm_model)

print(transcription['text'])
print(stutter_results)
```

---

## 📁 Project Structure

```
Multimodal-stuttering-analysis/
├── 📂 data/
│   ├── SEP-28k_labels.csv      # Dataset labels
│   └── clips/                   # Audio clips organized by source
│
├── 📂 models/
│   └── WaveLm_model.py         # WavLM classification model architecture
│
├── 📂 training/
│   ├── Dataset.py              # PyTorch Dataset for loading audio
│   ├── train_waveLM.py         # Training script
│   └── inference_wavelm.py     # WavLM inference utilities
│
├── 📂 inference/
│   ├── whisper_inference.py    # Whisper transcription
│   ├── wavlm_inference.py      # WavLM stutter detection
│   ├── predict.py              # Combined prediction pipeline
│   └── clinical_report.py      # Report generation
│
├── 📂 Preprocessing/
│   ├── audio_check.py          # Audio format validation
│   └── chunking.py             # Audio chunking utilities
│
├── 📂 evaluation/
│   └── metrics.py              # Evaluation metrics
│
├── 📂 checkpoints/
│   └── wavlm_stutter_classification_best.pth  # Trained model weights
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🌍 Real-World Impact & Use Cases

### 1. **Clinical Decision Support** 🏥
Speech-language pathologists can use this tool to:
- Get objective measurements of stuttering frequency
- Track patient progress over time
- Identify specific stutter types for targeted therapy

### 2. **Telehealth & Remote Assessment** 📱
- Enables stuttering assessment without in-person visits
- Patients can record and submit audio from home
- Bridges the gap in underserved areas lacking specialists

### 3. **Research & Data Collection** 📊
- Provides consistent, reproducible measurements
- Enables large-scale studies on stuttering patterns
- Supports development of new therapeutic approaches

### 4. **Self-Monitoring Tools** 👤
- Individuals who stutter can track their own progress
- Awareness of stutter patterns aids in self-management
- Reduces anxiety through objective feedback

### 5. **Educational Applications** 📚
- Training tool for speech pathology students
- Demonstrates stuttering types with real examples
- Interactive learning platform

---

## 🔬 Technical Innovation

| Aspect | Innovation |
|--------|------------|
| **Multimodal Approach** | Combines acoustic analysis (WavLM) with linguistic analysis (Whisper) |
| **Transfer Learning** | Leverages 94M pre-trained parameters, fine-tunes only 7.5% |
| **Real-time Capable** | Chunking enables processing of any length audio |
| **Multi-label Detection** | Detects multiple simultaneous stutter types |
| **Word-level Alignment** | Precise timestamps for each detected event |

---

## 📈 Future Roadmap

- [ ] Real-time streaming analysis
- [ ] Mobile app deployment
- [ ] Support for additional languages
- [ ] Severity scoring algorithm
- [ ] Integration with therapy platforms
- [ ] Longitudinal progress tracking

---

## 🙏 Acknowledgments

- **SEP-28k Dataset** - For providing labeled stuttering data
- **Microsoft Research** - For the WavLM pre-trained model
- **OpenAI** - For the Whisper speech recognition model
- **HuggingFace** - For the Transformers library

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Live Demo

<!-- ADD YOUR DEMO LINK BELOW -->
**[🚀 Try the Live Demo](YOUR_DEMO_LINK_HERE)**

Experience the Multimodal Stuttering Analysis System through our interactive web interface. Upload your audio (up to 30 seconds) and receive instant analysis.

---

## 📬 Contact

**Abhisar Gautam**

- GitHub: [@abhicodes-here2001](https://github.com/abhicodes-here2001)

---

<p align="center">
  <i>Built with the intent to make speech therapy assessments accessible to all 🗣️</i>
</p>
