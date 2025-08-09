# PyTorch Transformer for English-Twi Translation

A PyTorch implementation of a transformer model for English to Twi language translation with web interface, speech capabilities, and interactive notebooks.

## Dataset

Download the dataset from: https://www.kaggle.com/datasets/azunre/twi-dataset

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Required packages (see `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

Key dependencies include:
- PyTorch ecosystem (torch, torchvision, torchaudio, torchtext)
- Hugging Face libraries (datasets, tokenizers)
- ML tools (tensorboard, wandb, torchmetrics)
- Web interface (flask, flask-cors)
- Speech capabilities (openai-whisper, gTTS)
- Scientific computing (numpy, scipy)

## Project Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pryyyynz/pytorch-transformer
   cd pytorch-transformer
   ```

2. **Download the dataset**
   - Download from the Kaggle link above
   - Place the English and Twi text files in the `datasets/` directory:
     ```
     datasets/
     ├── english
     └── twi
     ```

3. **Clean and preprocess the data**
   ```bash
   python clean_dataset.py \
     --en-input datasets/english \
     --tw-input datasets/twi \
     --en-output datasets/english_clean.txt \
     --tw-output datasets/twi_clean.txt \
     --max-length 350 \
     --min-length 1 \
     --val-split 0.1
   ```

   This will:
   - Remove empty lines and normalize text
   - Filter sentences by length (max 350 tokens)
   - Create train/validation splits (90%/10%)
   - Save cleaned files to `datasets/`

4. **Train the model**
   
   Choose one of the training options:
   
   **Option A: Standard Training**
   ```bash
   python train.py
   ```
   
   **Option B: Training with Weights & Biases Logging**
   ```bash
   python train_wb.py
   ```

   The training script will:
   - Build tokenizers for both languages
   - Create the transformer model
   - Train for 30 epochs (configurable in `config.py`)
   - Save checkpoints to `custom_en_twi_weights/`
   - Log training metrics to TensorBoard (train.py) or Weights & Biases (train_wb.py)

## Training Options

### 1. Local Training (train.py)
Standard training with TensorBoard logging:
```bash
python train.py
```

### 2. Weights & Biases Training (train_wb.py)
Enhanced training with W&B experiment tracking:
```bash
python train_wb.py
```
This version includes:
- Comprehensive experiment logging
- Hyperparameter tracking
- Model performance visualization
- Online experiment dashboard

### 3. Jupyter Notebook Training
Interactive training and experimentation:
- `Local_Train.ipynb` - Local environment training
- `Colab_Train.ipynb` - Google Colab training
- `Inference.ipynb` - Model inference and testing
- `attention_visual.ipynb` - Attention mechanism visualization
- `Beam_Search.ipynb` - Beam search implementation

## Configuration

Edit `config.py` to adjust training parameters:
- `batch_size`: 16 (adjust based on GPU memory)
- `num_epochs`: 30
- `lr`: 5e-5 (learning rate)
- `seq_len`: 350 (maximum sequence length)
- `d_model`: 512 (model dimension)

## Web Interface

The project includes a Flask-based web application with speech capabilities:

### Features
- **Text Translation**: English to Twi text translation
- **Speech-to-Text**: Upload audio files for transcription using OpenAI Whisper
- **Text-to-Speech**: Generate audio from translated text using Google TTS
- **Interactive UI**: Modern web interface with real-time translation

### Running the Web Interface

1. **Start the Flask server**
   ```bash
   cd UI
   python app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to: `http://localhost:5005`

### Web Interface Features
- Upload audio files (wav, mp3, mp4, m4a, ogg, webm) for speech-to-text
- Real-time text translation
- Download translated text as audio
- Responsive design for mobile and desktop

### API Endpoints
- `POST /translate` - Translate text
- `POST /speech-to-text` - Convert audio to text
- `POST /text-to-speech` - Convert text to audio
- `GET /health` - Health check

## Inference

After training the model, you can use `translate.py` to translate text from English to Twi. The script uses the latest trained model checkpoint from `custom_en_twi_weights/` and the saved tokenizers.

### Running translate.py

The translation script supports three modes:

#### 1. Interactive Mode (Default)
Run without arguments for an interactive translation session:
```bash
python translate.py
```
- Enter English text when prompted
- Type `quit` to exit
- Type `random` to see random validation examples

#### 2. Command Line Mode
Translate a specific sentence directly from the command line:
```bash
python translate.py "Hello, how are you?"
```
Output:
```
Source: Hello, how are you?
Translation: [Twi translation]
```

#### 3. Random Validation Testing
Test the model on random sentences from the validation set:
```bash
# Translate 5 random validation sentences (default)
python translate.py --random

# Translate 10 random validation sentences
python translate.py --random 10
```
This shows the source (English), target (ground truth Twi), and predicted translations.

### Requirements for Inference

Before running `translate.py`, ensure you have:
1. Trained model weights in `custom_en_twi_weights/` (e.g., `tmodel_20.pt`)
2. Tokenizer files: `tokenizer_en.json` and `tokenizer_tw.json`
3. (Optional) Validation data in `datasets/` for random testing

### Python API Usage

You can also use the translation functionality in your own Python code:
```python
from translate import load_model_and_tokenizers, translate_sentence

# Load the model once
model, tokenizer_src, tokenizer_tgt, config, device = load_model_and_tokenizers()

# Translate multiple sentences
sentences = ["Hello world", "How are you?", "Good morning"]
for sentence in sentences:
    translation = translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, config, device)
    print(f"{sentence} -> {translation}")
```

## Interactive Development

### Jupyter Notebooks
The project includes several Jupyter notebooks for interactive development:

1. **Local_Train.ipynb** - Complete training pipeline for local environment
2. **Colab_Train.ipynb** - Google Colab optimized training
3. **Inference.ipynb** - Model inference and testing examples
4. **attention_visual.ipynb** - Attention mechanism visualization
5. **Beam_Search.ipynb** - Beam search decoding implementation

### Environment Setup
For conda users, use the provided environment file:
```bash
conda create --name transformer --file conda.txt
conda activate transformer
```

## Model Architecture

- Transformer architecture with:
  - 6 encoder/decoder layers
  - 8 attention heads
  - 512 model dimension
  - 2048 feedforward dimension
  - Dropout: 0.1

## File Structure

```
pytorch-transformer/
├── config.py           # Training configuration
├── train.py           # Main training script (TensorBoard)
├── train_wb.py        # Training with Weights & Biases
├── model.py           # Transformer model implementation
├── dataset.py         # Dataset and data loading
├── clean_dataset.py   # Data preprocessing script
├── translate.py       # Command-line inference script
├── requirements.txt   # Python dependencies
├── conda.txt         # Conda environment specification
├── datasets/          # Raw and cleaned data
├── custom_en_twi_weights/  # Model checkpoints
├── tokenizer_*.json   # Saved tokenizers
├── cache/            # Model and tokenizer cache
├── UI/               # Web interface
│   ├── app.py        # Flask application
│   ├── translate.py  # Translation utilities
│   ├── static/       # CSS and JavaScript
│   ├── templates/    # HTML templates
│   └── temp_audio/   # Temporary audio files
├── Local_Train.ipynb     # Local training notebook
├── Colab_Train.ipynb     # Google Colab notebook
├── Inference.ipynb      # Inference examples
├── attention_visual.ipynb # Attention visualization
└── Beam_Search.ipynb    # Beam search implementation
```

## Monitoring Training

### TensorBoard (train.py)
```bash
tensorboard --logdir=runs
```

### Weights & Biases (train_wb.py)
Visit your W&B dashboard after starting training to monitor:
- Loss curves
- Validation metrics
- Hyperparameter tracking
- Model performance comparisons

## Speech Features

### Speech-to-Text
- Powered by OpenAI Whisper
- Supports multiple audio formats
- Automatic transcription for translation input

### Text-to-Speech
- Uses Google Text-to-Speech (gTTS)
- Generates audio for translated text
- Fallback to English voice for Twi text

## Performance Tips

- Use mixed precision training for faster training and lower memory usage
- Adjust `batch_size` based on your GPU memory
- For GPUs with <16GB VRAM, reduce `batch_size` to 8 or 4
- Enable gradient accumulation for effective larger batch sizes

## Citation

If you use this code, please cite the original Twi dataset:
```
@dataset{azunre_twi_dataset,
  author = {Azunre, Paul},
  title = {Twi Dataset},
  year = {2021},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/azunre/twi-dataset}
}
```