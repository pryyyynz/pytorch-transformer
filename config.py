from pathlib import Path


def get_config():
    return {
        "batch_size": 16,  # Increased from 8 for more stable training
        "num_epochs": 30,  # Increased from 20 to allow more learning time
        "lr": 5*10**-5,    # Reduced from 10**-4 for better handling of specialized vocabulary
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'custom_en_twi',  # Changed from 'opus_books'
        "lang_src": "en",
        "lang_tgt": "tw",  # Changed from 'it' to 'tw'
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
