from flask import Flask, render_template, request, jsonify
import requests
import base64
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
import os
import random
import csv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Retrieve API key and endpoints from environment variables
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_TTS_ENDPOINT = os.getenv('AZURE_TTS_ENDPOINT')
AZURE_TTS_API_URL = os.getenv('AZURE_TTS_API_URL')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = 'tdaflaskmodel'  # Hardcoded deployment name


# Define the path to the resources folder
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), 'resources')

# Load idioms from CSV files
def load_idioms(filename):
    file_path = os.path.join(RESOURCE_PATH, filename)
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        idioms = [row[0] for row in reader if row]
    return idioms

# Load idioms from each category
advanced_idioms = load_idioms('advanced_idioms.csv')
genz_idioms = load_idioms('genz_idioms.csv')
business_idioms = load_idioms('business_idioms.csv')


def get_access_token():
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_API_KEY,
    }
    response = requests.post(AZURE_TTS_ENDPOINT, headers=headers)
    response.raise_for_status()
    return response.text


def generate_speech(text, access_token, language_code, voice_name, style=None):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/ssml+xml; charset=utf-8',
        'X-Microsoft-OutputFormat': 'audio-16khz-32kbitrate-mono-mp3'
    }
    if style:
        body = f"""
        <speak version='1.0' xml:lang='{language_code}'>
            <voice xml:lang='{language_code}' xml:gender='Female' name='{voice_name}' style='{style}'>
                {text}
            </voice>
        </speak>
        """
    else:
        body = f"""
        <speak version='1.0' xml:lang='{language_code}'>
            <voice xml:lang='{language_code}' xml:gender='Female' name='{voice_name}'>
                {text}
            </voice>
        </speak>
        """
    response = requests.post(AZURE_TTS_API_URL, headers=headers, data=body.encode('utf-8'))

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        response.raise_for_status()
    else:
        try:
            # Log the generated text and its length
            print(f"Generated Text: '{text}'")
            print(f"Generated Text Length: {len(text)}")

            # Attempt to decode audio and check if it is valid
            audio_data = response.content
            print(f"Received Audio Data Length: {len(audio_data)} bytes")

            if not audio_data or len(audio_data) < 100:  # Arbitrary small length check
                raise ValueError("Received audio data is invalid or too short.")

            return audio_data
        except Exception as e:
            print(f"Error decoding audio data: {str(e)}")
            raise


def generate_encouraging_message(language):
    if language == 'en':
        prompt = "Give me something encouraging to me, like the lover is praising me. I am tired of keeping working hard. Shorter, and like intimate conversation, talking style. Within 100 characters."
    else:  # zh
        prompt = "给我一些鼓励的话，就像恋人夸奖我一样。我厌倦了不断努力工作。短一点，就像亲密的对话一样。100个字符以内。"

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_API_KEY,
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an assistant that provides encouraging messages."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100  # Adjust max tokens to ensure shorter response
    }
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        message = response.json()["choices"][0]["message"]["content"].strip()
        if len(message) > 100:  # Additional check to truncate message if necessary
            message = message[:100]
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")

    return message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shadow', methods=['POST'])
def shadow():
    text = request.form['text']
    try:
        access_token = get_access_token()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error obtaining access token: {str(e)}'}), 500

    try:
        audio_content = generate_speech(text, access_token, 'en-US', 'en-US-JennyNeural')
        combined = AudioSegment.empty()
        for _ in range(5):
            audio_segment = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            combined += audio_segment

        # Convert the combined audio to base64
        buffered = BytesIO()
        combined.export(buffered, format="mp3")
        audio_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error generating speech: {str(e)}'}), 500

    return jsonify({'audio': audio_base64})

@app.route('/generate_sentences', methods=['POST'])
def generate_sentences():
    vocab = request.form['vocab']
    words = vocab.split(',')
    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_API_KEY,
    }

    prompt = f"Create a sentence using the following words: {', '.join(words)}"
    data = {
        "messages": [
            {"role": "system", "content": "You are an assistant that creates sentences using provided words."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        sentence = response.json()["choices"][0]["message"]["content"].strip()
    else:
        return jsonify({"error": response.json()}), response.status_code

    try:
        access_token = get_access_token()
        audio_content = generate_speech(sentence, access_token, 'en-US', 'en-US-JennyNeural')
        combined = AudioSegment.empty()
        for _ in range(5):
            audio_segment = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            combined += audio_segment

        # Convert the combined audio to base64
        buffered = BytesIO()
        combined.export(buffered, format="mp3")
        audio_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error generating speech: {str(e)}'}), 500

    return jsonify({'sentence': sentence, 'audio': audio_base64})

@app.route('/encourage', methods=['POST'])
def encourage():
    try:
        language = random.choice(['en', 'zh'])
        message = generate_encouraging_message(language)
        access_token = get_access_token()

        if language == 'en':
            language_code = 'en-US'
            voice_name = 'en-US-AriaNeural'
        else:  # zh
            language_code = 'zh-CN'
            voice_name = 'zh-CN-XiaoxiaoNeural'

        audio_content = generate_speech(message, access_token, language_code, voice_name, style='whispering')
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': message, 'audio': audio_base64})


def generate_idiom(category):
    if category == 'advanced':
        idiom = random.choice(advanced_idioms)
    elif category == 'genz':
        idiom = random.choice(genz_idioms)
    elif category == 'business':
        idiom = random.choice(business_idioms)
    else:
        raise ValueError("Invalid category")

    return idiom


def generate_example_sentence(idiom):
    prompt = f"Provide a sentence using the idiom '{idiom}' in a meaningful context."

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_API_KEY,
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an assistant that creates example sentences using provided idioms."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        response_data = response.json()
        print(f"Full Gen AI Response for Example Sentence: {response_data}")
        example_sentence = response_data["choices"][0]["message"]["content"].strip()
        return example_sentence
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")


def generate_idiom_meaning(idiom):
    # Prompt for generating the meaning in both English and Japanese
    prompt = f"Provide the meaning of the idiom '{idiom}' in simple English and Japanese."

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_API_KEY,
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an assistant that explains the meaning of idioms in both English and Japanese."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        response_data = response.json()
        print(f"Full Gen AI Response for Meaning: {response_data}")
        meaning = response_data["choices"][0]["message"]["content"].strip()

        return meaning
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")


@app.route('/get_idiom', methods=['POST'])
def get_idiom():
    category = request.form.get('idiom-category')
    if category not in ['advanced', 'genz', 'business']:
        return jsonify({"error": "Invalid category"}), 400

    try:
        # Generate idiom by randomly selecting from CSV files
        idiom = generate_idiom(category)
        print(f"Selected Idiom: {idiom}")

        # Generate meaning and example sentence as before
        meaning = generate_idiom_meaning(idiom)
        print(f"Generated Meaning: {meaning}")
        example_sentence = generate_example_sentence(idiom)
        print(f"Generated Example Sentence: {example_sentence}")

        if not example_sentence or len(example_sentence.split()) < 5:
            raise ValueError("Generated example sentence is too short or empty.")

        access_token = get_access_token()
        audio_content = generate_speech(example_sentence, access_token, 'en-US', 'en-US-JennyNeural')
        combined = AudioSegment.empty()
        for _ in range(5):
            audio_segment = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            combined += audio_segment

        buffered = BytesIO()
        combined.export(buffered, format="mp3")
        audio_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    except Exception as e:
        error_message = f"Error generating speech: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message}), 500

    return jsonify({'idiom': idiom, 'meaning': meaning, 'example': example_sentence, 'audio': audio_base64})


if __name__ == '__main__':
    app.run(debug=True)
