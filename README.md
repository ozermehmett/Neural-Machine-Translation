# Neural Machine Translation with GRU

This project is a Neural Machine Translation system based on GRU (Gated Recurrent Unit). It's developed using TensorFlow/Keras and features a Gradio web interface.

## 🚀 Features

* GRU (Gated Recurrent Unit) based encoder-decoder architecture
* TensorFlow/Keras implementation
* Gradio web interface
* Configurable config.py

## 📋 Requirements

```bash
tensorflow
keras
gradio
numpy
```

## 🌟 Example Usage

https://github.com/user-attachments/assets/08c23522-e005-4dc6-b39b-0d735d8a49e1

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ozermehmett/Neural-Machine-Translation.git
cd Neural-Machine-Translation
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Docker Installation (Optional)

```bash
# Build
docker build -t nmt-project .

# Run
docker run -p 7860:7860 nmt-project
```

## 💻 Usage

### Training the Model

```bash
python train.py
```

Note: Training parameters can be adjusted in the `config.py` file.

### Using the Gradio Interface

```bash
python app.py
```

Access the interface in your browser at `http://0.0.0.0:7860`

## ⚙️ Configuration (config.py)

```python
# Model parameters
class Config:
    BATCH_SIZE = 16
    MAX_LENGTH = 10
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 256

    EPOCHS = 10
    LEARNING_RATE = 0.001

    SOS_token = 0
    EOS_token = 1

    MODEL_PATH = "models/weights"
    DATA_PATH = "data"
````

## 🐳 Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

## 🔍 Troubleshooting

Common issues and solutions:
1. Memory error: Reduce batch size in config.py
2. OOM error: Decrease model size or input length
3. Port conflict: Use a different port number

## ⚡ Performance Tips

* Optimize batch size according to your system
* Adjust GRU units based on your data
* Use gradient clipping
* Monitor memory usage during training

## ✉️ Contact

Mehmet Özer
* GitHub: @ozermehmett

## 💡 Tips

* Review config.py before changing training parameters
* Increase batch size for larger datasets
* Adjust max_length parameter for longer sentences
* Use GPU acceleration when available

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies
3. Configure parameters in config.py
4. Run training script
5. Launch Gradio interface

## ⚠️ Requirements

* Python 3.7+
* CUDA-capable GPU (optional but recommended)
* Minimum 4GB VRAM or 8GB RAM
* Sufficient disk space for model checkpoints
