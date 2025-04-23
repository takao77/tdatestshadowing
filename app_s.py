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
from datetime import datetime, timezone, timedelta
import pyodbc
import bcrypt
import re
import time
import json
import difflib
import string
import math
from flask_mail import Mail, Message
import secrets


# Load environment variables from .env file
load_dotenv()


def hash_password(plain_password):
    plain_bytes = plain_password.encode('utf-8')
    hashed = bcrypt.hashpw(plain_bytes, bcrypt.gensalt())
    return hashed.decode('utf-8')  # 文字列に変換

def check_password(plain_password, hashed_password):
    plain_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_bytes, hashed_bytes)


def calculate_ef_with_decay(
    ef_old: float,
    self_score: int,
    repetitions: int,
    days_since: float,
    lam: float = 0.1,
    alpha: float = 0.3,
    beta: float = 0.1,
    perfect_boost: float = 2.0,
    min_ef: float = 0.8
) -> float:
    """
    時間減衰付き EF 更新モデル（改訂版）

    ef_old        : 前回の EF
    self_score    : 自己評価 (0～5)
    repetitions   : これまでのレビュー回数
    days_since    : 前回レビューからの経過日数
    lam           : 減衰率 λ
    alpha, beta   : 自己評価・反復回数の重み
    perfect_boost : 自己評価5時にEFを増加, 1時は逆数で減少
    min_ef        : EF の下限

    1) S = self_score / 5
    2) R = ln(1 + repetitions)
    3) D = exp(-lam * days_since)
    4) EF_base = ef_old * D + α·S + β·R
    5) self_score==5 のとき EF_base *= perfect_boost
       self_score==1 のとき EF_base /= perfect_boost
    6) EF_new = max(min_ef, EF_base)
    """
    S = self_score / 5.0
    R = math.log1p(repetitions)
    D = math.exp(-lam * days_since)

    ef_base = ef_old * D + alpha * S + beta * R

    if self_score == 5:
        ef_base *= perfect_boost
    elif self_score == 1:
        ef_base /= perfect_boost

    return max(min_ef, ef_base)


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


# ── メール設定 ─────────────────────────────────────────
app.config.update(
    MAIL_SERVER   = os.getenv('SMTP_HOST', 'smtp.sendgrid.net'),
    MAIL_PORT     = 587,
    MAIL_USE_TLS  = True,
    MAIL_USERNAME = os.getenv('SMTP_USER', 'apikey'),
    MAIL_PASSWORD = os.getenv('SG_API_KEY'),          # ★ 環境変数名は自由
    MAIL_DEFAULT_SENDER = (
        'TDA App',
        os.getenv('SMTP_FROM')
    )
)
mail = Mail(app)
VERIFY_BASE_URL = os.getenv(
    'VERIFY_BASE_URL',
    'https://tdatestshadowing.azurewebsites.net/verify_email'
)
# ────────────────────────────────────────────────

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
        user_id_input = request.form.get('user_id')
        password_input = request.form.get('password')

        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("""
            SELECT id, password_hash, name, is_email_verified
              FROM dbo.users
             WHERE user_id = ?
        """, user_id_input)
        row = cur.fetchone(); cur.close(); conn.close()

        if not row:
            return render_template('login.html', message='ユーザーが存在しません')

        db_id, db_pw_hash, db_name, verified = row
        if not verified:
            return render_template('login.html', message='メール認証を完了してください')

        if not check_password(password_input, db_pw_hash):
            return render_template('login.html', message='パスワードが違います')

        session['user_id'] = db_id
        session['user_name'] = db_name
        return redirect(url_for('home'))

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
        return jsonify({'error': 'User not logged in'}), 403

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
        SELECT 
          vr.review_time,
          vi.course,
          vi.sentence,
          vi.word,
          vr.self_score,
          vr.test_score,
          vr.ef,
          vr.next_review
        FROM dbo.vocab_reviews vr
        JOIN dbo.vocab_items vi 
          ON vr.vocab_id = vi.id
        WHERE vr.user_id = ?
        ORDER BY vr.review_time DESC
    """
    cursor.execute(sql, user_id)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    logs = []
    for row in rows:
        logs.append({
            'date':         row.review_time.isoformat(),
            'course':       row.course,
            'sentence':     row.sentence,
            'word':         row.word,
            'self_score':   row.self_score,
            'test_score':   row.test_score,
            'ef':           row.ef,
            'next_review':  row.next_review.isoformat() if row.next_review else ''
        })

    return jsonify(logs)


@app.route('/history')
def show_history():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('history.html')


@app.route('/get_shadow_sentence', methods=['GET'])
def get_shadow_sentence():
    # 1) 認証チェック
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 403
    user_id = session['user_id']

    # 2) クエリパラメータから course を取得
    selected_course = request.args.get('course', '')

    try:
        # 3) SQL から sentence と word を取得
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sentence, word
              FROM dbo.vocab_items
             WHERE course = ?
            """,
            selected_course
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # 4) データがなければ 404
        if not rows:
            return jsonify({'error': 'No sentences found for this course'}), 404

        # 5) ランダムに１件選択
        sentence, word = random.choice(rows)

        # 6) ログ保存（DB版） — 今回は word も渡す
        log_practice_activity_db(user_id, selected_course, sentence, word)

        # 7) JSON で返却
        return jsonify({'sentence': sentence, 'word': word})

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
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    conn = get_db_connection()
    cur  = conn.cursor()

    # courses と categories を JOIN し overview も取得
    cur.execute("""
        SELECT
            c.name               AS course_name,
            ISNULL(cat.name,
                   CASE
                        WHEN CHARINDEX('_', c.name) > 0
                             THEN LEFT(c.name, CHARINDEX('_', c.name) - 1)
                        ELSE c.name
                   END)           AS category_name,
            ISNULL(c.overview, '') AS overview
        FROM dbo.courses AS c
        LEFT JOIN dbo.categories AS cat
               ON c.category_id = cat.id
        WHERE c.is_public = 1
           OR c.owner_user_id = ?
        ORDER BY category_name, c.id ASC
    """, session['user_id'])

    rows = cur.fetchall()
    cur.close()
    conn.close()

    courses = [
        {
            "name":      r.course_name,
            "category":  r.category_name,
            "overview":  r.overview         # ← 追加
        }
        for r in rows
    ]

    return jsonify({"courses": courses})


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


@app.route('/api/get_due_vocab', methods=['GET'])
def get_due_vocab():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'ログインしてください'}), 403

    course = request.args.get('course', '')
    if not course:
        return jsonify({'error': 'course パラメータが必要です'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
            SELECT
                vi.id        AS vocab_id,
                vi.sentence,
                vi.word,
                ISNULL(vr.ef, 2.5) AS ef            -- ★ 追加: 最新 EF が無ければ 2.5
            FROM dbo.vocab_items  vi
            LEFT JOIN (
                SELECT vocab_id, ef, next_review, review_time,
                       ROW_NUMBER() OVER (PARTITION BY vocab_id
                                          ORDER BY review_time DESC) AS rn
                FROM dbo.vocab_reviews
                WHERE user_id = ?
            ) vr ON vi.id = vr.vocab_id AND vr.rn = 1           -- ★ ef を取得
            WHERE vi.course = ?
              AND (
                    vr.review_time IS NULL
                 OR vr.review_time <= DATEADD(minute,-10,GETUTCDATE())
              )
            ORDER BY ISNULL(vr.next_review,'1900-01-01') ASC
        """, user_id, course)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    due = [{
        'vocab_id': row.vocab_id,
        'sentence': row.sentence,
        'word':     row.word,
        'ef': float(row.ef)
    } for row in rows]

    return jsonify(due)


@app.route('/submit_practice', methods=['POST'])
def submit_practice():
    user_id    = session.get('user_id')
    vocab_id   = request.form.get('vocab_id', type=int)
    self_score = request.form.get('self_score', type=int)

    if not user_id or not vocab_id or self_score is None:
        return jsonify({'error': 'Missing parameters'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # --- 1) 最新レビューを取得 ---
    cursor.execute("""
        SELECT TOP 1
           ef,
           review_time,
           COUNT(*) OVER (PARTITION BY user_id, vocab_id) AS reps
        FROM dbo.vocab_reviews
        WHERE user_id = ? AND vocab_id = ?
        ORDER BY review_time DESC
    """, user_id, vocab_id)
    row = cursor.fetchone()

    if row:
        prev_ef    = row.ef or 2.5
        last_time  = row.review_time
        reps       = row.reps + 1
        days_since = max((datetime.utcnow() - last_time).days, 0)
    else:
        prev_ef    = 2.5   # 初回 EF
        reps       = 1
        days_since = 0

    # --- 2) EF を計算 ---
    new_ef = calculate_ef_with_decay(
        ef_old=prev_ef,
        self_score=self_score,
        repetitions=reps,
        days_since=days_since
    )

    # --- 3) 次回レビュー日時を決定 ---
    next_review = datetime.utcnow() + timedelta(days=new_ef)

    # --- 4) レビュー履歴を INSERT ---
    cursor.execute("""
        INSERT INTO dbo.vocab_reviews
          (user_id, vocab_id, review_time, self_score, ef, next_review)
        VALUES (?, ?, ?, ?, ?, ?)
    """, user_id, vocab_id, datetime.utcnow(), self_score, new_ef, next_review)

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        'ef': round(new_ef, 2),
        'next_review': next_review.isoformat()
    })
from flask import abort


# ───────────────────────────────────────────────────────
# 管理画面ページを表示
@app.route('/admin/courses')
def admin_courses():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('admin_courses.html')


# ───────────────────────────────────────────────────────
# (1) 既存コース一覧取得
@app.route('/api/get_courses_admin', methods=['GET'])
def get_courses_admin():
    # 1) ログインチェック
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    # 2) セッションから user_id を直接取得（整数型）
    owner_user_id = session['user_id']

    # 3) DB からそのユーザー所有のコースを取得
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute(
        """
        SELECT
            c.id,
            c.name,
            c.language,
            c.is_public,
            c.category_id,
            COALESCE(cat.name, '') AS category_name
        FROM dbo.courses AS c
        LEFT JOIN dbo.categories AS cat
               ON c.category_id = cat.id
        WHERE c.owner_user_id = ?
        ORDER BY c.id
        """,
        owner_user_id
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # ── 3) JSON 整形 ────────────────────────────────
    courses = [
        {
            'id':            r.id,
            'name':          r.name,
            'language':      r.language,
            'is_public':     bool(r.is_public),
            'category_id':   r.category_id,
            'category_name': r.category_name
        }
        for r in rows
    ]

    return jsonify({'courses': courses})



# ───────────────────────────────────────────────────────
# (2) コース追加
@app.route('/api/create_course', methods=['POST'])
def create_course():
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    data      = request.get_json() or {}
    name      = data.get('name', '').strip()
    language  = data.get('language', '').strip()
    is_public = 1 if data.get('is_public') else 0
    category_id = int(data.get('category_id', 0) or 0)  # 追加
    if category_id is None:
        return jsonify({'error': 'category_id が必要です'}), 400

    if not name or not language:
        return jsonify({'error': 'パラメータ不足'}), 400

    owner_user_id = session['user_id']
    conn = get_db_connection()
    cur  = conn.cursor()

    # INSERT と同時に新規作成された ID を取得
    cur.execute("""
        INSERT INTO dbo.courses (name, language, is_public, owner_user_id, category_id)
        OUTPUT INSERTED.id
        VALUES (?, ?, ?, ?, ?)
    """, name, language, is_public, owner_user_id, category_id)

    row = cur.fetchone()
    if not row:
        conn.rollback()
        cur.close()
        conn.close()
        return jsonify({'error': 'コース作成に失敗しました'}), 500

    new_id = row[0]  # OUTPUT INSERTED.id の結果

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'success': True, 'id': new_id}), 201


# (3) コース更新（公開／非公開切替など）
@app.route('/api/update_course', methods=['POST'])
def update_course():
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    data = request.get_json() or {}
    course_id = data.get('course_id')
    name      = data.get('name')
    language  = data.get('language')
    is_public = data.get('is_public')
    category_id = int(data.get('category_id', 0) or 0)

    if course_id is None or is_public is None:
        return jsonify({'error': 'パラメータ不足'}), 400

    owner_user_id = session['user_id']

    conn = get_db_connection()
    cur  = conn.cursor()

    # 部分更新として、渡ってきたフィールドのみ更新
    updates = []
    params  = []
    if name is not None:
        updates.append("name = ?")
        params.append(name.strip())
    if language is not None:
        updates.append("language = ?")
        params.append(language.strip())
    if category_id is not None:
        updates.append("category_id = ?")
        params.append(category_id)
    # 公開フラグは必須
    updates.append("is_public = ?")
    params.append(1 if is_public else 0)

    # WHERE 用パラメータ（course_id, owner_user_id）
    params.extend([course_id, owner_user_id])

    sql = f"""
        UPDATE dbo.courses
           SET {', '.join(updates)}
         WHERE id = ? AND owner_user_id = ?
    """
    cur.execute(sql, *params)
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({}), 200


# (4) コース削除
@app.route('/api/delete_course/<int:course_id>', methods=['DELETE'])
def delete_course(course_id):
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    owner_user_id = session['user_id']

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        DELETE FROM dbo.courses
         WHERE id = ? AND owner_user_id = ?
    """, course_id, owner_user_id)
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({}), 200


@app.route('/api/get_categories')
def get_categories():
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id, name FROM dbo.categories ORDER BY id")
    rows = cur.fetchall()
    cur.close(); conn.close()

    cats = [{'id': r.id, 'name': r.name} for r in rows]
    return jsonify({'categories': cats})


@app.route('/admin/vocab_upload')
def vocab_upload_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('vocab_upload.html')


# app_s.py などに追記
@app.route('/api/upload_vocab', methods=['POST'])
def upload_vocab():
    if 'user_id' not in session:
        return jsonify({'error': 'ログインしてください'}), 403

    file = request.files.get('csv_file')
    if not file or file.filename == '':
        return jsonify({'error': 'CSV ファイルを選択してください'}), 400

    import io, csv
    # ★ UTF‑8 or UTF‑8‑BOM どちらも正しく読めるよう utf‑8‑sig にする
    decoded = io.TextIOWrapper(file.stream, encoding='utf-8-sig', newline='')
    reader  = csv.DictReader(decoded)

    conn = get_db_connection()
    cur  = conn.cursor()

    # ★ 1) 先に courses テーブル全件を読んで set にしておく
    cur.execute("SELECT name FROM dbo.courses")
    valid_courses = {row.name.strip() for row in cur.fetchall()}

    inserted, errors = 0, []

    for lineno, row in enumerate(reader, start=2):   # 1 行目はヘッダ
        course   = (row.get('course')   or '').strip()
        sentence = (row.get('sentence') or '').strip()
        word     = (row.get('word')     or '').strip()

        # ★ 2) コースが存在しなければエラーログに記録してスキップ
        if course not in valid_courses:
            errors.append(f'{lineno} 行目: コース「{course}」は存在しません')
            continue

        if not (course and sentence):
            errors.append(f'{lineno} 行目: 必須列が足りません')
            continue

        try:
            cur.execute("""
                INSERT INTO dbo.vocab_items (course, sentence, word)
                VALUES (?, ?, ?)
            """, course, sentence, word or None)
            inserted += 1
        except Exception as e:
            # ★ DB 制約違反などのエラーを収集
            errors.append(f'{lineno} 行目: {e}')

    conn.commit()
    cur.close(); conn.close()

    return jsonify({'inserted': inserted, 'errors': errors})


# ────────────────────── ユーザー登録 ──────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    # POST: フォーム値取得
    user_id = request.form.get('user_id', '').strip()
    name    = request.form.get('name', '').strip()
    email   = request.form.get('email', '').strip()
    pw      = request.form.get('password')

    # 簡易バリデーション
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return render_template('register.html', msg='無効なメールアドレスです')

    # パスワードハッシュ
    pw_hash = hash_password(pw)

    # メール確認用トークン
    token = secrets.token_urlsafe(32)

    # DB へ仮登録
    conn, cur = get_db_connection(), None
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dbo.users
                (user_id, name, password_hash, email,
                 is_email_verified, verify_token)
            VALUES (?, ?, ?, ?, 0, ?)
        """, user_id, name, pw_hash, email, token)
        conn.commit()
    except pyodbc.IntegrityError:
        return render_template('register.html', msg='User ID または Email が既に存在します')
    finally:
        if cur: cur.close()
        conn.close()

    # 確認メール送信
    verify_link = f'{VERIFY_BASE_URL}?token={token}'
    body = f'''{name} さん

以下のリンクをクリックしてメールアドレスを確認してください。

{verify_link}

TDA App'''
    mail.send(Message(
        subject='【TDA】メールアドレス確認のお願い',
        recipients=[email],
        body=body
    ))
    return render_template('register_done.html')

# ────────────────────── メール確認リンク ──────────────────────
@app.route('/verify_email')
def verify_email():
    token = request.args.get('token', '')
    conn  = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        UPDATE dbo.users
           SET is_email_verified = 1,
               verify_token      = NULL
         WHERE verify_token = ?
    """, token)
    updated = cur.rowcount
    conn.commit(); cur.close(); conn.close()

    msg = ('認証に成功しました。ログインできます。' if updated
           else 'リンクが無効、または期限切れです。')
    return render_template('verify_result.html', msg=msg)


if __name__ == '__main__':
    app.run(debug=True)
