import os
import sys
import random
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import boto3
import torch
from tokenizers import Tokenizer

# Add parent directory to Python path to import config and model
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Now import the model and config after adding the parent directory to path
from config import get_config, latest_weights_file_path
from model import build_transformer

# Load environment variables
load_dotenv()


class S3ModelLoader:
    """Handles downloading and caching models from S3"""

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, s3_key):
        """Generate a cache file path for an S3 key"""
        # Use hash of S3 key to create a unique filename
        key_hash = hashlib.md5(s3_key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}_{Path(s3_key).name}"

    def _is_cached(self, s3_key):
        """Check if file is already cached locally"""
        cache_path = self._get_cache_path(s3_key)
        if cache_path.exists():
            try:
                # Check if the cached file is newer than 24 hours
                # You can adjust this logic based on your needs
                import time
                file_age = time.time() - cache_path.stat().st_mtime
                return file_age < 86400  # 24 hours in seconds
            except:
                return False
        return False

    def download_file(self, s3_key, force_download=False):
        """Download file from S3 with caching"""
        cache_path = self._get_cache_path(s3_key)

        if not force_download and self._is_cached(s3_key):
            print(f"Using cached file: {cache_path}")
            return cache_path

        try:
            print(f"Downloading {s3_key} from S3...")
            self.s3_client.download_file(
                self.bucket_name, s3_key, str(cache_path))
            print(f"Downloaded and cached: {cache_path}")
            return cache_path
        except Exception as e:
            if cache_path.exists():
                print(
                    f"Download failed, using existing cached file: {cache_path}")
                return cache_path
            raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")

    def list_s3_files(self, prefix=""):
        """List files in S3 bucket with optional prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Error listing S3 files: {e}")
            return []


def load_model_and_tokenizers():
    """Load the model and tokenizers from S3 with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config()
    s3_loader = S3ModelLoader()

    try:
        # Define S3 keys for the files
        tokenizer_src_key = "tokenizer_en.json"
        tokenizer_tgt_key = "tokenizer_tw.json"
        model_key = "tmodel_20.pt"

        # Download tokenizers from S3
        print("Loading tokenizers from S3...")
        tokenizer_src_path = s3_loader.download_file(tokenizer_src_key)
        tokenizer_tgt_path = s3_loader.download_file(tokenizer_tgt_key)

        # Load tokenizers
        try:
            tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
            tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizers: {e}")

        # Build model
        model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            config["seq_len"],
            config['seq_len'],
            d_model=config['d_model']
        ).to(device)

        # Download and load model weights from S3
        print("Loading model weights from S3...")
        model_path = s3_loader.download_file(model_key)

        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            print(f"Model loaded successfully from S3: {model_key}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights from S3: {e}")

        return model, tokenizer_src, tokenizer_tgt, config, device

    except Exception as e:
        # Fallback to local files if S3 fails
        print(f"S3 loading failed: {e}")
        print("Attempting to load from local files...")
        return load_model_and_tokenizers_local()


def load_model_and_tokenizers_local():
    """Fallback function to load model and tokenizers from local files"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config()

    # Check if tokenizer files exist
    tokenizer_src_path = Path(
        config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(
        config['tokenizer_file'].format(config['lang_tgt']))

    if not tokenizer_src_path.exists():
        raise FileNotFoundError(
            f"Source tokenizer file not found: {tokenizer_src_path}")
    if not tokenizer_tgt_path.exists():
        raise FileNotFoundError(
            f"Target tokenizer file not found: {tokenizer_tgt_path}")

    try:
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizers: {e}")

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(
            "No model weights found. Please train the model first.")

    try:
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully from {model_filename}")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")

    return model, tokenizer_src, tokenizer_tgt, config, device


def translate_sentence(sentence: str, model, tokenizer_src, tokenizer_tgt, config, device):
    """Translate a single sentence"""
    seq_len = config['seq_len']

    # Check if sentence is empty
    if not sentence.strip():
        return ""

    with torch.no_grad():
        # Encode the source sentence
        source = tokenizer_src.encode(sentence)
        source_ids = source.ids

        # Check if sentence is too long
        if len(source_ids) > seq_len - 2: # -2 for SOS and EOS tokens
            print(
                f"Warning: Input sentence is too long ({len(source_ids)} tokens). Truncating to fit sequence length.")
            source_ids = source_ids[:seq_len - 2]

        # Build the source tensor with special tokens and padding
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                         dtype=torch.int64),
            torch.tensor(source_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                         dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                         (seq_len - len(source_ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        source = source.unsqueeze(0)  # Add batch dimension
        source_mask = (source != tokenizer_src.token_to_id(
            '[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)

        # Encode
        encoder_output = model.encode(source, source_mask)

        # Initialize decoder input with SOS token
        decoder_input = torch.empty(1, 1).fill_(
            tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Generate translation
        while decoder_input.size(1) < seq_len:
            # Create causal mask for decoder
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(
                1), decoder_input.size(1))), diagonal=1).type(torch.int)
            decoder_mask = (decoder_mask == 0).type_as(source_mask).to(device)

            # Decode
            out = model.decode(encoder_output, source_mask,
                               decoder_input, decoder_mask)

            # Get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
                source).fill_(next_word.item()).to(device)], dim=1)

            # Stop if EOS token is generated
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

        # Decode the translation
        translation = tokenizer_tgt.decode(decoder_input[0].tolist())
        # Clean up the translation by removing special tokens
        translation = translation.replace('[SOS]', '').replace(
            '[EOS]', '').replace('[PAD]', '').strip()

    return translation


def load_validation_data():
    """Load validation data from cleaned files"""
    val_en_path = Path('datasets/english_clean_val.txt')
    val_tw_path = Path('datasets/twi_clean_val.txt')

    if not val_en_path.exists() or not val_tw_path.exists():
        print("Validation files not found. Please run clean_dataset.py first.")
        return None, None

    try:
        with open(val_en_path, 'r', encoding='utf-8') as f:
            english_sentences = f.readlines()

        with open(val_tw_path, 'r', encoding='utf-8') as f:
            twi_sentences = f.readlines()

        return english_sentences, twi_sentences
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None, None


def translate_random_validation(model, tokenizer_src, tokenizer_tgt, config, device, num_samples=5):
    """Randomly select and translate validation sentences"""
    english_sentences, twi_sentences = load_validation_data()

    if english_sentences is None:
        return

    # Ensure we have sentences to translate
    if len(english_sentences) == 0:
        print("No validation sentences found.")
        return

    # Randomly select indices
    indices = random.sample(range(len(english_sentences)), min(
        num_samples, len(english_sentences)))

    print(f"\nTranslating {len(indices)} random validation sentences:\n")
    print("-" * 80)

    for idx in indices:
        source = english_sentences[idx].strip()
        target = twi_sentences[idx].strip()

        try:
            translation = translate_sentence(
                source, model, tokenizer_src, tokenizer_tgt, config, device)

            print(f"Source (EN): {source}")
            print(f"Target (TW): {target}")
            print(f"Predicted:   {translation}")
            print("-" * 80)
        except Exception as e:
            print(f"Error translating sentence at index {idx}: {e}")
            print("-" * 80)


def interactive_mode(model, tokenizer_src, tokenizer_tgt, config, device):
    """Interactive translation mode"""
    print("\nInteractive Translation Mode")
    print("Type 'quit' to exit, 'random' to translate random validation sentences")
    print("-" * 80)

    while True:
        try:
            text = input("\nEnter English text to translate: ").strip()

            if text.lower() == 'quit':
                break
            elif text.lower() == 'random':
                translate_random_validation(
                    model, tokenizer_src, tokenizer_tgt, config, device)
            elif text:
                translation = translate_sentence(
                    text, model, tokenizer_src, tokenizer_tgt, config, device)
                print(f"Translation: {translation}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during translation: {e}")


def main():
    """Main function"""
    try:
        # Load model and tokenizers once
        model, tokenizer_src, tokenizer_tgt, config, device = load_model_and_tokenizers()

        if len(sys.argv) > 1:
            if sys.argv[1] == '--random':
                # Translate random validation sentences
                num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
                translate_random_validation(
                    model, tokenizer_src, tokenizer_tgt, config, device, num_samples)
            else:
                # Translate provided sentence
                sentence = ' '.join(sys.argv[1:])
                translation = translate_sentence(
                    sentence, model, tokenizer_src, tokenizer_tgt, config, device)
                print(f"Source: {sentence}")
                print(f"Translation: {translation}")
        else:
            # Interactive mode
            interactive_mode(model, tokenizer_src,
                             tokenizer_tgt, config, device)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
