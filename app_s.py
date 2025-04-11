from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import azure.cognitiveservices.speech as speechsdk
import requests
import base64
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
import os
import random
import csv
from datetime import datetime
import pyodbc
import time
import json
import difflib
import string


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY', 'default_secret_key')

# Retrieve API key and endpoints from environment variables
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_TTS_ENDPOINT = os.getenv('AZURE_TTS_ENDPOINT')
AZURE_TTS_API_URL = os.getenv('AZURE_TTS_API_URL')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = 'tda_4'  # Hardcoded deployment name
AZURE_SPEECH_API_KEY = os.getenv('AZURE_SPEECH_API_KEY')  # For Speech API
AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')  # Region for Speech API

# Define the path to the resources folder
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), 'resources')
USER_FILE = os.path.join(RESOURCE_PATH, 'users.csv')  # Path to the user data CSV

# Define the path to the user logs folder
USER_LOGS_PATH = os.path.join(os.path.dirname(__file__), 'user_logs')
os.makedirs(USER_LOGS_PATH, exist_ok=True)

# Get the absolute path to the resources folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_FOLDER = os.path.join(BASE_DIR, 'resources')

# Path to the 'user_audio' folder
USER_AUDIO_PATH = os.path.join(os.path.dirname(__file__), 'user_audio')

# Ensure the 'user_audio' folder exists
os.makedirs(USER_AUDIO_PATH, exist_ok=True)


def get_db_connection():
    """
    Azure Web App のアプリ設定 `AZURE_SQL_CONNECTION_STRING` を使ってDB接続を返す。
    """
    conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("AZURE_SQL_CONNECTION_STRING is not set or empty.")

    # pyodbcで接続
    conn = pyodbc.connect(conn_str)
    return conn


def log_practice_activity_db(user_id, course, sentence, word):
    conn = None
    cursor = None
    """
    practice_logs テーブルにログをINSERTする。
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 例: INSERT
        sql = """
        INSERT INTO practice_logs (log_datetime, user_id, course, sentence, word)
        VALUES (?, ?, ?, ?, ?)
        """
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(sql, now_str, user_id, course, sentence, word)
        conn.commit()

        print("Inserted log to practice_logs successfully.")
    except Exception as e:
        print(f"Error inserting log to DB: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def log_practice_activity(user_id, course, sentence):
    """
    Appends a new practice record to practice_logs.csv in the resources folder.
    Storing only DateTime, UserID, Course, and Sentence.
    added on 6.4.2025
    """
    practice_log_path = os.path.join(RESOURCE_PATH, 'practice_logs.csv')
    file_exists = os.path.isfile(practice_log_path)

    # Current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define the CSV headers
    fieldnames = ['DateTime', 'UserID', 'Course', 'Sentence']

    # Open the CSV file in append mode
    with open(practice_log_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If it's a brand new file, write the header first
        if not file_exists:
            writer.writeheader()

        # Write the row
        writer.writerow({
            'DateTime': timestamp,
            'UserID': user_id,
            'Course': course,
            'Sentence': sentence
        })


def log_user_activity(user_id, idiom, example_sentence):
    log_file_path = os.path.join(USER_LOGS_PATH, f'{user_id}.csv')
    file_exists = os.path.isfile(log_file_path)

    # Get current date in day/month/year format
    current_date = datetime.now().strftime('%d/%m/%Y')

    with open(log_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Date', 'User ID', 'Idiom', 'Example Sentence'])
        writer.writerow([current_date, user_id, idiom, example_sentence])

def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 2:
                    user_id, name, password = row
                    users[user_id] = {'name': name, 'password': password}
    return users

USERS = load_users()

def check_user_credentials(user_id, password):
    user = USERS.get(user_id)
    if user and user['password'] == password:
        return user['name']
    return None

# Load idioms from CSV files
def load_idioms(filename):
    file_path = os.path.join(RESOURCE_PATH, filename)
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        idioms = [row[0] for row in reader if row]
    return idioms

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
            print(f"Generated Text: '{text}'")
            print(f"Generated Text Length: {len(text)}")

            audio_data = response.content
            print(f"Received Audio Data Length: {len(audio_data)} bytes")

            if not audio_data or len(audio_data) < 100:
                raise ValueError("Received audio data is invalid or too short.")

            return audio_data
        except Exception as e:
            print(f"Error decoding audio data: {str(e)}")
            raise

def generate_encouraging_message(language):
    if language == 'en':
        prompt = "Give me something encouraging to me, like the lover is praising me. I am tired of keeping working hard. Shorter, and like intimate conversation, talking style. Within 100 characters."
    else:
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
        "max_tokens": 100
    }
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        message = response.json()["choices"][0]["message"]["content"].strip()
        if len(message) > 100:
            message = message[:100]
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")

    return message

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
        example_sentence = response_data["choices"][0]["message"]["content"].strip()
        return example_sentence
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")

def generate_idiom_meaning(idiom):
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
        meaning = response_data["choices"][0]["message"]["content"].strip()
        return meaning
    else:
        raise Exception(f"Error from OpenAI: {response.status_code}, {response.json()}")


@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')

        if user_id and password:
            # Assuming check_user_credentials validates and returns the user's name
            user_name = check_user_credentials(user_id, password)

            if user_name:
                # Store user information in the session
                session['user_id'] = user_id
                session['user_name'] = user_name

                # Redirect to shadowing page after successful login
                return redirect(url_for('home'))

        # If login fails, display the error message on the login page
        return render_template('login.html', message='IDとパスワードを入れてください')

    # Render login page for GET request
    return render_template('login.html')


@app.route('/home')
def home():
    # Ensure the user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login_user'))  # Redirect to login if not authenticated

    # Render the home.html page for course selection
    return render_template('home.html', user_name=session.get('user_name'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_user'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('shadowing.html', user_name=session.get('user_name'))

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
        else:
            language_code = 'zh-CN'
            voice_name = 'zh-CN-XiaoxiaoNeural'

        audio_content = generate_speech(message, access_token, language_code, voice_name, style='whispering')
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': message, 'audio': audio_base64})

@app.route('/get_idiom', methods=['POST'])
def get_idiom():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403

    user_id = session['user_id']
    category = request.form.get('idiom-category')
    if category not in ['advanced', 'genz', 'business']:
        return jsonify({"error": "Invalid category"}), 400

    try:
        idiom = generate_idiom(category)
        meaning = generate_idiom_meaning(idiom)
        example_sentence = generate_example_sentence(idiom)

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


@app.route('/user_logs', methods=['GET'])
def user_logs():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403

    user_id = session['user_id']

    # cursor変数を先に宣言しておく
    conn = None
    cursor = None

    try:
        conn = get_db_connection()  # ← ここで失敗する場合がある
        cursor = conn.cursor()  # ← ここに到達しないと cursor が未定義

        # SQL実行の例
        sql = """
        SELECT log_datetime, user_id, course, sentence, word
        FROM practice_logs
        WHERE user_id = ?
        ORDER BY log_datetime DESC
        """
        cursor.execute(sql, user_id)
        rows = cursor.fetchall()

        logs = []
        for row in rows:
            # 変更後 (wordも追加)
            logs.append({
                'date': str(row.log_datetime),
                'user_id': row.user_id,
                'course': row.course,
                'sentence': row.sentence,
                'word': row.word
            })

        return jsonify(logs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # cursor や conn が None でない場合のみ close する
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/history')
def show_history():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('history.html')


@app.route('/get_shadow_sentence', methods=['GET'])
def get_shadow_sentence():
    # CSVファイルへのパス
    shadow_csv_path = os.path.join(RESOURCES_FOLDER, 'shadow.csv')

    # 1) セッションでユーザがログインしているか確認
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403

    # 2) ログイン中のユーザIDを取り出す
    user_id = session['user_id']

    # 3) クエリパラメータ course=? で指定されたコース名を取得
    selected_course = request.args.get('course', '')

    sentences = []
    entries = []

    try:
        # 4) shadow.csv を読み込み、course列が選択したコースに一致するものを集める
        with open(shadow_csv_path, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row['course'] == selected_course:
                    entries.append({
                        'sentence': row.get('sentence', ''),
                        'word': row.get('word', ''),  # ★ 追加
                        'translated': row.get('translated', ''),
                        'tips': row.get('tips', ''),
                        'sequence': row.get('sequence', ''),
                    })

        if entries:
            selected_entry = random.choice(entries)

            # 6) ログを追記する（CSVまたはDBなど）
            log_practice_activity_db(user_id, selected_course, selected_entry['sentence'], selected_entry['word'])

            # 7) 選んだ文章をJSONで返す
            return jsonify({
                'sentence': selected_entry['sentence'],
                'word': selected_entry['word'],  # ★ 追加
                'translated': selected_entry['translated'],
                'tips': selected_entry['tips'],
                'sequence': selected_entry['sequence']
            })
        else:
            return jsonify({'error': 'No sentences found for the selected course'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_shadow_audio', methods=['POST'])
def generate_shadow_audio():
    sentence = request.form['sentence']
    mode = request.form.get('mode', 'practice')  # Default mode is 'practice'
    try:
        access_token = get_access_token()  # Assuming this function is defined to get Azure access
        audio_content = generate_speech(sentence, access_token, 'en-US', 'en-US-JennyNeural')

        combined = AudioSegment.empty()

        if mode == 'practice':
            # Play the audio 5 times for Practice Mode
            for _ in range(5):
                audio_segment = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
                combined += audio_segment
        else:
            # Play the audio once for Record Mode
            audio_segment = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            combined = audio_segment

        # Export combined audio and return as base64
        buffered = BytesIO()
        combined.export(buffered, format="mp3")
        audio_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'audio': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shadowing', methods=['GET'])
def shadowing():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))

    # Get the selected course from the query parameters
    selected_course = request.args.get('course', '')

    # Pass the selected course to the shadowing.html page
    return render_template('shadowing.html', user_name=session.get('user_name'), course=selected_course)



@app.route('/save_recording', methods=['POST'])
def save_recording():
    """Saves the user's recorded audio as a .wav file."""

    # Check if the user is logged in
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403

    # Get the user's ID and name from the session
    user_id = session['user_id']
    user_name = session['user_name']

    # Create a folder for the user if it doesn't exist
    user_folder = os.path.join(USER_AUDIO_PATH, user_name)
    os.makedirs(user_folder, exist_ok=True)

    # Get the uploaded audio data from the request
    audio_data = request.files.get('audio_data')

    if not audio_data:
        return jsonify({"error": "No audio data provided"}), 400

    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_filename = f'{timestamp}.wav'

    # Path to save the audio file
    audio_path = os.path.join(user_folder, audio_filename)

    # Save the audio file to the user's folder
    audio_data.save(audio_path)

    return jsonify({"success": True, "message": "Audio saved successfully", "audio_path": audio_path})


def ensure_correct_wav_format(input_file):
    # Convert the audio file to WAV format with the correct specifications
    audio = AudioSegment.from_file(input_file)
    wav_output_path = input_file.replace(".webm", ".wav")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_output_path, format="wav")
    return wav_output_path


# Function to assess pronunciation
@app.route('/assess_pronunciation', methods=['POST'])
def assess_pronunciation():
    """Performs pronunciation assessment on a saved audio file."""

    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403

    audio_path = request.form.get('audio_path')
    sentence = request.form.get('sentence')

    if not audio_path or not sentence:
        return jsonify({"error": "Missing audio path or sentence"}), 400

    if not os.path.exists(audio_path):
        return jsonify({"error": f"Audio file not found: {audio_path}"}), 400

    try:
        # Convert to proper WAV format if necessary
        audio_path = ensure_correct_wav_format(audio_path)

        # Initialize the SpeechConfig and AudioConfig with the WAV file
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_API_KEY, region=AZURE_SPEECH_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

        # Set up pronunciation assessment config
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=sentence,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
        )

        # Create a speech recognizer using the WAV file as audio input
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        pronunciation_config.apply_to(speech_recognizer)

        # Recognize once (simpler flow)
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
            return jsonify({
                "success": True,
                "result": {
                    "pronunciationScore": pronunciation_result.pronunciation_score,
                    "accuracyScore": pronunciation_result.accuracy_score,
                    "completenessScore": pronunciation_result.completeness_score,
                    "fluencyScore": pronunciation_result.fluency_score
                }
            })
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return jsonify({"success": False, "error": "No speech could be recognized."})
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            return jsonify({"success": False, "error": f"Speech Recognition canceled: {cancellation_details.reason}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/get_courses', methods=['GET'])
def get_courses():
    course_csv_path = os.path.join(RESOURCES_FOLDER, 'course.csv')
    courses = []

    try:
        with open(course_csv_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if 'course' in row:  # Ensure 'course' column exists
                    courses.append(row['course'])

        return jsonify({"courses": courses})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate_explanation', methods=['POST'])
def generate_explanation():
    """
    Receives a sentence and returns a Japanese explanation
    with details for English learners. Uses Azure OpenAI.
    """
    try:
        sentence = request.form.get('sentence', '').strip()
        word = request.form.get('word', '').strip()
        if not sentence:
            return jsonify({"error": "No sentence provided."}), 400

        prompt = f"""
1. 以下の英文: 
\"{sentence}\"
について、日本語の訳を述べる
２．次に
単語 '{word}'
について、語源や覚えるためのコツを語学学習者のために解説してください。
"""

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_API_KEY,
        }

        data = {
            "messages": [
                {"role": "system", "content": "You are an assistant that explains English sentences in Japanese for learners."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400
        }

        # POSTリクエスト
        import requests
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            response_data = response.json()
            explanation_text = response_data["choices"][0]["message"]["content"].strip()
            return jsonify({"explanation": explanation_text})
        else:
            return jsonify({"error": f"OpenAI Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_explanation_cn', methods=['POST'])
def generate_explanation_cn():
    try:
        sentence = request.form.get('sentence', '').strip()
        word = request.form.get('word', '').strip()
        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        # 中国語向けのPrompt
        prompt = f"""
        以下的英文：
        「{sentence}」

        1. 请先用中文进行翻译，并对整个句子做一个通俗易懂的解释。
        2. 对其中的单词「{word}」，请说明它的词源、记忆方法或相关有趣故事，以帮助学习者更好地掌握这个单词。

        请用友好、易于理解的口吻来回答，并尽量提供详细的说明。
        """

        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_API_KEY,
        }

        data = {
            "messages": [
                {"role": "system",
                 "content": "You are an assistant that explains English sentences in Japanese for learners."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400
        }

        # POSTリクエスト
        import requests
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            response_data = response.json()
            explanation_text = response_data["choices"][0]["message"]["content"].strip()
            return jsonify({"explanation": explanation_text})
        else:
            return jsonify({"error": f"OpenAI Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
