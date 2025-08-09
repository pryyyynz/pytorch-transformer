import torch
import whisper
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from translate import load_model_and_tokenizers, translate_sentence
import sys
import os
from gtts import gTTS
import uuid
import time

# Add the parent directory to the path to find model-related files
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now import from translate after adding parent directory to path

# Create a directory for temporary audio files
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'temp_audio')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Global variables to store model and tokenizers
model = None
tokenizer_src = None
tokenizer_tgt = None
config = None
device = None
whisper_model = None

# Configure upload settings
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'm4a', 'ogg', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_model():
    """Initialize the model and tokenizers"""
    global model, tokenizer_src, tokenizer_tgt, config, device, whisper_model
    try:
        # Change to parent directory where the model files are located
        original_dir = os.getcwd()
        os.chdir(parent_dir)

        print("Loading translation model and tokenizers...")
        model, tokenizer_src, tokenizer_tgt, config, device = load_model_and_tokenizers()
        print("Translation model loaded successfully!")

        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully!")

        # Change back to original directory
        os.chdir(original_dir)
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        # Change back to original directory in case of error
        try:
            os.chdir(original_dir)
        except:
            pass
        return False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests"""
    global model, tokenizer_src, tokenizer_tgt, config, device

    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if model is None:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500

        # Translate the text
        translation = translate_sentence(
            text, model, tokenizer_src, tokenizer_tgt, config, device)

        return jsonify({
            'success': True,
            'original': text,
            'translation': translation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Handle speech-to-text requests using Whisper"""
    global whisper_model

    try:
        # Check if whisper model is loaded
        if whisper_model is None:
            return jsonify({'error': 'Whisper model not loaded. Please restart the server.'}), 500

        # Check if the post request has the file part
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        if file and allowed_file(file.filename):
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                file.save(temp_file.name)

                try:
                    # Transcribe the audio using Whisper
                    result = whisper_model.transcribe(temp_file.name)
                    transcribed_text = result["text"].strip()

                    return jsonify({
                        'success': True,
                        'text': transcribed_text
                    })

                except Exception as e:
                    return jsonify({'error': f'Error transcribing audio: {str(e)}'}), 500

                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
        else:
            return jsonify({'error': 'Invalid file type. Supported formats: wav, mp3, mp4, m4a, ogg, webm'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Handle text-to-speech requests using Google TTS"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        # Default to Twi, but allow specification
        language = data.get('language', 'tw')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Map language codes to gTTS language codes
        language_map = {
            'en': 'en',   # English
            # Use English (Ghana) as closest approximation for Twi
            'tw': 'en-gh'
        }

        # Get the correct language code for gTTS
        # Default to English (Ghana) if not found
        tts_lang = language_map.get(language, 'en-gh')

        try:
            # Generate a unique filename for the audio file
            audio_filename = os.path.join(
                TEMP_AUDIO_DIR, f"{uuid.uuid4()}.mp3")

            # Convert text to speech using Google TTS
            tts = gTTS(text=text, lang=tts_lang)
            tts.save(audio_filename)

            # Wait for the file to be saved
            time.sleep(1)

            # Send the audio file to the client
            return send_file(audio_filename, mimetype='audio/mpeg', as_attachment=True, download_name='speech.mp3')

        except ValueError as ve:
            # If language is not supported, try with English as fallback
            if "Language not supported" in str(ve):
                # Log the error
                print(
                    f"Language {tts_lang} not supported, falling back to English")

                # Generate speech with English voice
                tts = gTTS(text=text, lang='en')
                audio_filename = os.path.join(
                    TEMP_AUDIO_DIR, f"{uuid.uuid4()}.mp3")
                tts.save(audio_filename)
                time.sleep(1)

                return send_file(audio_filename, mimetype='audio/mpeg', as_attachment=True, download_name='speech.mp3')
            else:
                raise

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize model on startup
    if initialize_model():
        app.run(debug=True, host='0.0.0.0', port=5005)
    else:
        print("Failed to initialize model. Exiting.")
