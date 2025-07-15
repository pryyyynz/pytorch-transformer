# PyTorch Transformer for English-Twi Translation

A PyTorch implementation of a transformer model for English to Twi language translation.

## Dataset

Download the dataset from: https://www.kaggle.com/datasets/azunre/twi-dataset

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Required packages:
  ```bash
  pip install torch torchvision torchaudio
  pip install tokenizers datasets
  pip install tqdm tensorboard
  pip install pandas numpy
  ```

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
   ```bash
   python train.py
   ```

   The training script will:
   - Build tokenizers for both languages
   - Create the transformer model
   - Train for 30 epochs (configurable in `config.py`)
   - Save checkpoints to `custom_en_twi_weights/`
   - Log training metrics to TensorBoard

## Configuration

Edit `config.py` to adjust training parameters:
- `batch_size`: 16 (adjust based on GPU memory)
- `num_epochs`: 30
- `lr`: 5e-5 (learning rate)
- `seq_len`: 350 (maximum sequence length)
- `d_model`: 512 (model dimension)

## Training on Remote GPU

If training on a remote server:

1. **Upload files to server**
   ```bash
   scp -r . user@server:~/pytorch-transformer/
   ```

2. **SSH into server and run training**
   ```bash
   ssh user@server
   cd pytorch-transformer
   python train.py
   ```

3. **Download trained models**
   ```bash
   # From local machine
   scp user@server:~/pytorch-transformer/custom_en_twi_weights/*.pt ./custom_en_twi_weights/
   scp user@server:~/pytorch-transformer/tokenizer_*.json ./
   ```

## Inference

To translate text using the trained model:
```python
from translate import translate

# Load the model (uses latest checkpoint by default)
text = "Hello, how are you?"
translation = translate(text)
print(translation)
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
├── train.py           # Main training script
├── model.py           # Transformer model implementation
├── dataset.py         # Dataset and data loading
├── clean_dataset.py   # Data preprocessing script
├── tokenizer.py       # Tokenizer utilities
├── translate.py       # Inference script
├── datasets/          # Raw and cleaned data
├── custom_en_twi_weights/  # Model checkpoints
└── tokenizer_*.json   # Saved tokenizers
```

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

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