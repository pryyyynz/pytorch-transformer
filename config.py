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
    model_folder_path = Path(model_folder)
    
    # Check if the folder exists
    if not model_folder_path.exists():
        print(f"Weights folder '{model_folder}' does not exist")
        return None
    
    model_filename = f"{config['model_basename']}*.pt"
    weights_files = list(model_folder_path.glob(model_filename))
    
    if len(weights_files) == 0:
        print(f"No weight files found in '{model_folder}'")
        return None
    
    # Sort by extracting the epoch number from filename
    def get_epoch_number(filepath):
        # Extract number from filename like "tmodel_00.pt"
        filename = filepath.stem  # removes .pt extension
        epoch_str = filename.replace(config['model_basename'], '')
        try:
            return int(epoch_str)
        except ValueError:
            return -1
    
    weights_files.sort(key=get_epoch_number)
    latest_file = weights_files[-1]
    print(f"Found latest checkpoint: {latest_file}")
    return str(latest_file)
