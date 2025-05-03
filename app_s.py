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
from openai import AzureOpenAI
import io
import time
import json
import difflib
import string
import math
from flask_mail import Mail, Message
import secrets
import tempfile
import subprocess, tempfile, os, json, shlex
from collections import namedtuple


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
AZURE_OPENAI_STT_ENDPOINT =os.getenv('AZURE_OPENAI_STT_ENDPOINT')
AZURE_OPENAI_STT_KEY = os.getenv('AZURE_OPENAI_STT_KEY')
AZURE_OPENAI_STT_DEPLOY ="stt_model"
STT_API_VER  = "2024-02-15-preview"

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
        'Polyagent AI',
        os.getenv('SMTP_FROM')
    )
)
mail = Mail(app)
VERIFY_BASE_URL = os.getenv(
    'VERIFY_BASE_URL',
    'https://tdatestshadowing.azurewebsites.net/verify_email'
)
# ────────────────────────────────────────────────

client = AzureOpenAI(
    api_key      = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version  = "2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # 例: https://xxx.openai.azure.com
)

DEPLOY_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "tda_4")   # 4-o-mini のデプロイ名

# --------------------- ① Chat 用 ----------------------------
chat_client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),   # https://xxx.openai.azure.com
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = "2024-02-15-preview"
)
CHAT_DEPLOY = os.getenv("AZURE_OPENAI_CHAT_DEPLOY", "tda_4")    # gpt-4o-mini など

# --------------------- ② STT 用（Whisper） ------------------
stt_client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_STT_ENDPOINT"),    # 別リソースなら URL も別
    api_key        = os.getenv("AZURE_OPENAI_STT_KEY"),
    api_version    = "2024-02-15-preview"
)

STT_DEPLOY = os.getenv("AZURE_OPENAI_STT_DEPLOY", "stt_model")  # 例: whisper-1 の独自名

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


def fetchone(sql, *params):
    """1 行だけ返して接続を即クローズ。見つからなければ None"""
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute(sql, *params)
    row = cur.fetchone()
    cur.close(); conn.close()
    return row


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


# --- 既存ユーティリティの上に追記 or 置換 -----------------
DEFAULT_VOICE = "en-US-JennyMultilingualNeural"   # Long-Form 対応 voice
DEFAULT_STYLE = "chat"                            # 自然な会話調


def generate_speech(
    text: str,
    access_token: str,
    language_code: str = "en-US",
    voice_name: str   = DEFAULT_VOICE,
    style: str        = DEFAULT_STYLE
) -> bytes:
    """
    Azure Neural TTS で音声バイナリを返す。
    Long-Form Neural Voice + style 属性をデフォルトにした改訂版。
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/ssml+xml; charset=utf-8",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
    }

    # --- SSML を組み立て -----------------------------------
    ssml = f"""
    <speak version="1.0" xml:lang="{language_code}"
    xmlns:mstts="http://www.w3.org/2001/mstts">
      <voice name="{voice_name}">
        <mstts:express-as style="{style}">
          {text}
        </mstts:express-as>
      </voice>
    </speak>
    """

    url = os.getenv("AZURE_TTS_API_URL")
    response = requests.post(url, headers=headers, data=ssml.encode("utf-8"))
    response.raise_for_status()
    return response.content


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


# ────────────────────────────────────────────
# Azure Speech で WAV → テキスト
def speech_to_text(wav_path: str) -> str:
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_API_KEY,
        region      =AZURE_SPEECH_REGION
    )
    audio_config  = speechsdk.audio.AudioConfig(filename=wav_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config =audio_config
    )
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    raise RuntimeError(f'Speech‐to‐text failed: {result.reason}')


def tts_to_b64(
    text: str,
    voice: str = DEFAULT_VOICE,
    style: str = DEFAULT_STYLE,
    fmt: speechsdk.SpeechSynthesisOutputFormat =
         speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
) -> str:
    """
    Azure Speech SDK で TTS → base64。
    Long-Form voice + style がデフォルト。
    """
    speech_cfg = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION")
    )
    # 出力フォーマットを忘れず設定
    speech_cfg.set_speech_synthesis_output_format(fmt)

    # --- ★ 名前空間を明示した SSML -------------------------
    ssml = f"""
<speak version="1.0"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="{voice.split('-')[0]}">
  <voice name="{voice}">
    <mstts:express-as style="{style}">
      {text}
    </mstts:express-as>
  </voice>
</speak>
""".strip()

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_cfg,
        audio_config=None
    )
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        # 取消理由を詳しく取得
        details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        raise RuntimeError(f"TTS canceled: {details.reason} - {details.error_details}")

    return base64.b64encode(result.audio_data).decode("ascii")


# 返り値用の軽量コンテナ
LatestReview = namedtuple('LatestReview', 'ef reps days_since')

def get_latest_review(user_id: int, vocab_id: int):
    """
    指定ユーザー × 指定単語の直近レビュー情報を 1 行だけ取得して返す。

    戻り値:
        LatestReview(ef, reps, days_since)
        - ef         : 直近の EF (なければ 2.5)
        - reps       : 今回を含める前の累積レビュー回数
        - days_since : 直近レビューから現在までの日数
      もし履歴が無ければ None を返す。
    """
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT TOP 1
               ef,
               review_time,
               COUNT(*) OVER (PARTITION BY user_id, vocab_id) AS reps
          FROM dbo.vocab_reviews
         WHERE user_id = ? AND vocab_id = ?
         ORDER BY review_time DESC
    """, user_id, vocab_id)

    row = cur.fetchone()
    cur.close(); conn.close()

    if not row:                       # 未学習
        return None

    # 経過日数 (0 日未満にならないようガード)
    days = max((datetime.utcnow() - row.review_time).days, 0)

    return LatestReview(
        ef   = row.ef or 2.5,         # NULL のとき初期値 2.5
        reps = row.reps,              # 直近レビュー以前の回数
        days_since = days
    )


def execute(sql,*params):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute(sql,*params); conn.commit()
    cur.close(); conn.close()


def translate_word_to_jp(word: str) -> str:
    """
    OpenAI (Azure) で英単語を 1 語の日本語に翻訳して返す。
    失敗時は空文字列を返す。
    """
    prompt = f'次の英単語を日本語に翻訳して。およそ３語から５語で。さらに、この英単語の最初のアルファベットを書いてください。: "{word}"'
    url = (f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
           f"{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15")
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    data = {
        "messages": [
            {"role": "system",
             "content": "You are a bilingual assistant. Output ONLY the Japanese translation."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30,
        "temperature": 0.3
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        jp = r.json()["choices"][0]["message"]["content"].strip()
        return jp.split('\n')[0]      # 複数行返ってきたら先頭だけ
    except Exception as e:
        app.logger.warning("translate_word_to_jp failed: %s", e)
        return ""


# Azure OpenAI 定数は既存のものを再利用
CHAT_URL = (
    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
    f"{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15"
)
HEADERS  = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

def ai_pick_token(sentence: str, lemma: str) -> str | None:
    """
    GPT に「lemma が句動詞やイディオムなら完全形を返せ」と要求。
    戻り値が sentence に実在しない or 複数行の場合は None。
    """
    prompt = (
        "From the sentence below, return EXACTLY the token or phrase "
        "that corresponds to the given lemma.\n"
        "- If the lemma is part of a multi-word phrasal verb or idiom, "
        "return the *whole phrase* as it appears in the sentence.\n"
        "- Output just that phrase. No extra words, no punctuation."
        f"\nSentence: {sentence}\nLemma: {lemma}"
    )
    payload = {
        "messages": [
            {"role":"system",
             "content":"You are a precise linguistic extractor. "
                       "Return only the matched phrase."},
            {"role":"user","content":prompt}
        ],
        "max_tokens": 10,"temperature":0
    }
    try:
        res = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=15)
        res.raise_for_status()
        phrase = res.json()["choices"][0]["message"]["content"].strip()
        # 改行・余計な語がないか最小チェック
        if "\n" in phrase or phrase.lower() not in sentence.lower():
            return None
        return phrase
    except Exception as e:
        app.logger.info("ai_pick_token error: %s", e)
        return None


PARTICLES = {"up","out","off","in","on","down","over","away","back","through"}

def _local_fallback(sentence:str, lemma:str)->str|None:
    """GPT が失敗した時用 ― 連続 n語 を類似度で探す"""
    lemma_lc = lemma.lower()
    l_parts  = lemma_lc.split()
    n        = len(l_parts)
    tokens   = re.findall(r"[A-Za-z']+", sentence)
    best, best_score = None, 0.0

    for i in range(len(tokens)-n+1):
        cand = ' '.join(tokens[i:i+n]).lower()
        sc   = difflib.SequenceMatcher(None, cand, lemma_lc).ratio()
        if sc > best_score:
            best, best_score = ' '.join(tokens[i:i+n]), sc

    return best if best_score >= 0.6 else None


def make_cloze(sentence: str, lemma: str) -> tuple[str,str]:
    """
    戻り値: (cloze_sentence, answer_phrase)
      - answer_phrase は実際に空欄にした語句
    """
    # ---------- ① GPT で語句を取得 ----------
    phrase = ai_pick_token(sentence, lemma)

    # ---------- ② GPT が1語のみ返したら粒子を自動追加 ----------
    if phrase and ' ' not in phrase:
        patt = re.compile(r'\b' + re.escape(phrase) + r'\b\s+(\w+)', re.IGNORECASE)
        m = patt.search(sentence)
        if m and m.group(1).lower() in PARTICLES:
            phrase = f"{phrase} {m.group(1)}"   # 例: fill out

    # ---------- ③ まだ見つからなければローカル fallback ----------
    if not phrase:
        phrase = _local_fallback(sentence, lemma)

    # ---------- ④ 置換 ----------
    if phrase:
        blanks = ' '.join('_'*len(w) for w in phrase.split())
        clz    = re.sub(re.escape(phrase), blanks, sentence, count=1, flags=re.IGNORECASE)
        return clz, phrase

    # 最後の安全策: 置換せずそのまま
    return sentence, lemma


def translate_en_to_jp(text: str) -> str:
    """
    Azure OpenAI で英文を自然な日本語に翻訳して返す。
    """
    payload = {
        "messages":[
            {"role":"system","content":"You are a professional translator. "
                                       "Translate the sentence into natural Japanese."},
            {"role":"user","content": text}
        ],
        "max_tokens": 120,
        "temperature": 0
    }
    r = requests.post(
        CHAT_URL,           # 既に定義済み (AZURE_OPENAI_ENDPOINT …)
        headers=HEADERS,    # 既に定義済み
        json=payload,
        timeout=15
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


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

        # ── ★ ④ ログイン成功 ⇒ verify_token を NULL にする ──
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
                UPDATE dbo.users
                   SET verify_token = NULL
                 WHERE id = ?
                   AND verify_token IS NOT NULL      -- 既に NULL ならスキップ
            """, db_id)
        conn.commit()
        cur.close();
        conn.close()

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
            ORDER BY
                    COALESCE(vr.ef, 3.0)               ASC,
                    ISNULL(vr.next_review,'1900-01-01') ASC,
                    NEWID()
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
Please click the link below to verify your e-mail address.

{verify_link}

Polyagent AI'''
    mail.send(Message(
        subject='[Polyagent AI] メール確認 / Please verify your e-mail',
        recipients=[email],
        body=body
    ))
    return render_template('register_done.html')

# ────────────────────── メール確認リンク ──────────────────────
# --- メール確認リンク ---------------------------
@app.route('/verify_email')
def verify_email():
    token = request.args.get('token', '').strip()

    if not token:
        return render_template(
            'verify_result.html',
            success=False,
            msg='リンクが無効、または期限切れです。/ The link is invalid or has expired.',
            updated_rows=0
        )

    conn = get_db_connection(); cur = conn.cursor()

    # ① token が一致する行を認証済みにする（token は消さない）
    cur.execute("""
        UPDATE dbo.users
           SET is_email_verified = 1
         WHERE verify_token      = ?
    """, token)
    updated = cur.rowcount          # 1 行なら今回初めて認証

    # ② 既に認証済みだったか（is_email_verified=1）が残っているか判定
    if updated == 0:
        cur.execute("""
            SELECT 1
              FROM dbo.users
             WHERE verify_token      = ?
               AND is_email_verified = 1
        """, token)
        if cur.fetchone():
            updated = 1             # すでに認証済みとみなす

    conn.commit(); cur.close(); conn.close()

    success = (updated == 1)
    msg = 'メールアドレスの確認が完了しました！/ Your e-mail has been verified successfully!' if success \
          else 'リンクが無効、または期限切れです。/ The link is invalid or has expired.'

    return render_template(
        'verify_result.html',
        success=success,
        msg=msg,
        updated_rows=updated   # デバッグ表示用
    )


# ─────────────────────────────────────────────
# 失効・未認証時にメールを再送するエンドポイント
# ─────────────────────────────────────────────
@app.route('/resend_verification', methods=['POST'])
def resend_verification():
    email = request.form.get('email', '').strip().lower()

    if not email:
        return jsonify({'ok': False, 'msg': 'Email is required'}), 400

    conn = get_db_connection()
    cur  = conn.cursor()
    # 未認証ユーザーを取得
    cur.execute("""
        SELECT id, user_id, name 
          FROM dbo.users 
         WHERE email = ? AND is_email_verified = 0
    """, email)
    row = cur.fetchone()

    if not row:
        cur.close(); conn.close()
        return jsonify({'ok': False, 'msg': 'User not found or already verified'}), 404

    user_id, user_name = row.user_id, row.name
    new_token = secrets.token_urlsafe(32)

    # トークン更新
    cur.execute("""
        UPDATE dbo.users
           SET verify_token = ?
         WHERE id = ?
    """, new_token, row.id)
    conn.commit()
    cur.close(); conn.close()

    # メール送信
    verify_url = f"{VERIFY_BASE_URL}?token={new_token}"
    mail.send(Message(
        subject="Welcome to Polyagent AI – Verify your email",
        recipients=[email],
        html=render_template('mail_verify.html',
                             user_name=user_name,
                             verify_url=verify_url)
    ))

    return jsonify({'ok': True, 'msg': 'Verification mail sent'})


@app.route("/api/stt_to_text", methods=["POST"])
def stt_to_text():
    """
    フロントから送られた wav バイナリを
    gpt-4o-mini-transcribe デプロイ (audio/transcriptions) へ投げ、
    プレーンテキストを返す。
    """
    try:
        # 受信 ------------------------------------------------------
        file_storage = request.files["audio"]  # Werkzeug FileStorage
        blob = file_storage.read()  # bytes
        mime = file_storage.mimetype or "application/octet-stream"
        filename = file_storage.filename or "audio.webm"  # 後方互換で名前も拝借
        # ---------- REST 呼び出し ----------
        url = (
            f"{AZURE_OPENAI_STT_ENDPOINT}/openai/deployments/{AZURE_OPENAI_STT_DEPLOY}"
            f"/audio/transcriptions?api-version={STT_API_VER}"
        )
        headers = {"api-key": AZURE_OPENAI_STT_KEY}
        files   = {"file": (filename, blob, mime)}
        data = {
            "response_format": "text"       # json ではなく text
            # language, prompt など必要に応じ追加
        }

        r = requests.post(url, headers=headers, files=files, data=data, timeout=90)
        r.raise_for_status()                # 200 以外は例外送出
        text = r.text.strip()               # text/plain で返る

        return jsonify({"text": text})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/stt_to_text_speech", methods=["POST"])
def stt_to_text_speech():
    """
    * WebM / WAV → そのまま Whisper v2
    * それ以外（mp4 など）→ ffmpeg で 16-kHz mono WAV 化して Whisper
      ── 変換に失敗したら 400 を返す
    Whisper 結果が英語以外なら 400
    """
    try:
        # ---------- 受信 ----------
        fs     = request.files["audio"]
        blob   = fs.read()
        mime   = fs.mimetype or "application/octet-stream"
        fname  = fs.filename or "speech_input"

        # ---------- ★ デバッグ用に保存・ログ出力する ---------- #
        from pathlib import Path
        import tempfile, os

        # ① ─── 保存先を OS に合わせて動的に決める ──────────
        #    Linux:  /tmp/last_ios_upload.mp4
        #    macOS:  /tmp/last_ios_upload.mp4
        #    Windows: %TEMP%\last_ios_upload.mp4  例) C:\Users\foo\AppData\Local\Temp
        dbg_path = Path(tempfile.gettempdir()) / "last_ios_upload.mp4"

        # ② ─── 親ディレクトリが無い場合は作る（念のため） ─────
        dbg_path.parent.mkdir(parents=True, exist_ok=True)

        # ③ ─── バイト列を書き出す ───────────────────────
        try:
            dbg_path.write_bytes(blob)
            app.logger.info("🛠 saved debug audio: %s (%d bytes)",
                            dbg_path, dbg_path.stat().st_size)
        except Exception as e:
            # 失敗しても本処理は続ける
            app.logger.warning("debug-save failed: %s", e)
        # ------------------------------------------------------

        app.logger.info("STT upload: mime=%s, size=%d", mime, len(blob))

        # ----- ★ デバッグ用に一時保存 -----------------------
        import uuid, pathlib, tempfile
        dbg_path = pathlib.Path(tempfile.gettempdir()) / f"dbg_{uuid.uuid4()}.bin"
        dbg_path.write_bytes(blob)             # ← ここ
        app.logger.info("Saved debug blob → %s", dbg_path)
        # ---------------------------------------------------

        # ---------- Whisper 呼び出しヘルパ ----------
        def call_whisper(file_tup):
            url = (f"{AZURE_OPENAI_STT_ENDPOINT.rstrip('/')}"
                   f"/openai/deployments/{AZURE_OPENAI_STT_DEPLOY}"
                   f"/audio/transcriptions?api-version={STT_API_VER}")
            return requests.post(
                url,
                headers={"api-key": AZURE_OPENAI_STT_KEY},
                files={"file": file_tup},
                data={"response_format": "text", "language": "en"},
                timeout=90
            )

        # ---------- ① まず無加工で投げる ----------
        resp = call_whisper((fname, BytesIO(blob), mime))
        if resp.status_code == 200:
            text = resp.text.strip()
            if not text or re.search(r"[\u3040-\u30ff\u4e00-\u9faf]", text):
                return jsonify({"error": "英語を話してくださいね！"}), 400
            return jsonify({"text": text})

        # ---------- ② model_error → WAV へ再試行 ----------
        if resp.status_code == 400 and "model_error" in resp.text:
            app.logger.warning("Whisper model_error – retry with WAV")

            try:
                # MIME を見て format を明示 (mp4 / m4a は mp4)
                fmt = "mp4" if mime in ("audio/mp4", "audio/m4a",
                                        "video/mp4") else None
                seg = (AudioSegment
                       .from_file(BytesIO(blob), format=fmt)  # ← ★ここ重要
                       .set_frame_rate(16000)
                       .set_channels(1)
                       .set_sample_width(2))
                wav_buf = BytesIO()
                seg.export(wav_buf, format="wav")
                wav_buf.seek(0)
            except Exception as e:
                app.logger.error("ffmpeg convert fail:\n%s", e)
                return jsonify({"error": "Audio conversion failed"}), 400

            resp = call_whisper(("speech.wav", wav_buf, "audio/wav"))
            resp.raise_for_status()

            text = resp.text.strip()
            if not text or re.search(r"[\u3040-\u30ff\u4e00-\u9faf]", text):
                return jsonify({"error": "英語を話してくださいね！"}), 400
            return jsonify({"text": text})

        # ---------- その他の 4xx 5xx ----------
        app.logger.warning("Whisper %d (%s)", resp.status_code, resp.text[:120])
        resp.raise_for_status()                # ここで例外化

    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai_chat", methods=["POST"])
def ai_chat():
    j = request.get_json(force=True)
    user_text   = j.get('user_text','')
    history     = j.get('history',[])
    target_word = j.get('target_word','')
    target_sentence = j.get('target_sentence', '')
    need_jp = j.get('need_jp_help',False)

    sys_prompt=f"""
You are Alex, an English tutor.

Target **word** : "{target_word}"
Target **sentence** : "{target_sentence}"

First, briefly explain the word *and* the sentence, then ask the learner
to repeat the sentence aloud. After the learner answers, score 1-5, etc…

NEED_JP_HELP={need_jp}
Respond JSON: {{"reply":..., "score":n, "jp_hint":...}}
"""

    messages=[{"role":"system","content":sys_prompt}] + \
             [{"role":h['role'],"content":h['text']} for h in history] + \
             [{"role":"user","content":user_text}]

    chat = chat_client.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=messages,
        temperature=0.7,
        response_format={"type":"json_object"}
    )
    jres=json.loads(chat.choices[0].message.content)
    ai_text = jres.get('reply','')
    score   = int(jres.get('score',3))
    jp_hint = jres.get('jp_hint','')

    # --- EF 更新
    # if target_word and 'vocab_id' in j:
    #     form={'vocab_id':j['vocab_id'],'self_score':score}
    #     requests.post(url_for('submit_practice',_external=True),data=form)

    ai_audio_b64 = tts_to_b64(ai_text)

    new_hist = (history+[{'role':'user','text':user_text},{'role':'assistant','text':ai_text}])[-10:]
    return jsonify({'ai_text':ai_text,'ai_audio':ai_audio_b64,'score':score,'jp_hint':jp_hint,'new_history':new_hist})


@app.route('/api/ai_session/start')
def ai_session_start():
    """返り値: { words:[{vocab_id,word,sentence,ef}], user_level:str }"""
    if 'user_id' not in session:
        return jsonify({'error':'login required'}),403

    course  = request.args.get('course','')
    user_id = session['user_id']

    conn,cursor = get_db_connection(),None
    try:
        cursor = conn.cursor()
        cursor.execute("""
            WITH latest AS (
              SELECT vocab_id, ef,
                     ROW_NUMBER()OVER(PARTITION BY vocab_id ORDER BY review_time DESC) rn
                FROM dbo.vocab_reviews WHERE user_id=?
            )
            SELECT TOP 3 vi.id, vi.word, vi.sentence,
                   ISNULL(lat.ef,2.5) AS ef
              FROM dbo.vocab_items vi
         LEFT JOIN latest lat ON vi.id = lat.vocab_id AND lat.rn=1
             WHERE vi.course=?
               AND ( lat.vocab_id IS NULL -- 未学習
                  OR lat.ef < 4           -- 要復習
                  OR lat.vocab_id IN (SELECT vocab_id FROM latest WHERE ef<4) )
             ORDER BY NEWID()            -- ランダム 3 語
        """, user_id, course)
        rows=[{'vocab_id':r.id,'word':r.word,'sentence':r.sentence,'ef':float(r.ef)} for r in cursor]
    finally:
        if cursor: cursor.close(); conn.close()

    level = 'beginner' if sum(w['ef'] for w in rows)/max(len(rows),1) < 2.2 else 'intermediate'
    return jsonify({'words':rows,'user_level':level})


@app.route('/api/sakura', methods=['POST'])
def sakura_teacher():
    data      = request.get_json(force=True)
    word      = data.get('word','').strip()
    sentence  = data.get('sentence','').strip()

    prompt = f"""
あなたは『さくら先生』というやさしい日本語教師です。
Word: 「{word}」
Sentence: 「{sentence}」

1. 単語を小学生にも分かるように簡単に解説
2. 例文の意味を解説
3. 単語について頻繁に使われるCollocationsコロケーションをいくつか紹介
4. 最後に相手を励ます短いメッセージ
口調は親しみやすく、語尾に♡などは使わず自然体で。
"""

    gpt = chat_client.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=[{"role":"system","content":prompt}],
        temperature=0.7
    )
    jp_text = gpt.choices[0].message.content.strip()

    # ——— TTS  (ja-JP-NanamiNeural + cheerful) ———
    access = get_access_token()
    audio  = generate_speech(
        text = jp_text,
        access_token = access,
        language_code="ja-JP",
        voice_name="ja-JP-NanamiNeural",
        style="cheerful"          # gentle & 明るい
    )
    b64 = base64.b64encode(audio).decode('utf-8')
    return jsonify({'text': jp_text, 'audio': b64})


@app.route('/api/review/next')
def get_next_review():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error':'login'}),403

    sql = """       -- EF 条件を外し “全語” 抽出
    WITH latest AS(
        SELECT vocab_id,ef,review_time,
               ROW_NUMBER()OVER(PARTITION BY vocab_id ORDER BY review_time DESC) rn
        FROM dbo.vocab_reviews WHERE user_id=?
    )
    SELECT TOP 1 vi.id,vi.word,vi.sentence,lat.ef
      FROM latest lat
      JOIN dbo.vocab_items vi ON vi.id=lat.vocab_id
     WHERE lat.rn=1
       AND lat.review_time<=DATEADD(minute,-10,GETUTCDATE())
     ORDER BY lat.ef ASC,lat.review_time ASC,NEWID()"""
    conn=get_db_connection(); cur=conn.cursor()
    cur.execute(sql,uid); row=cur.fetchone()
    cur.close(); conn.close()

    if not row:
        return jsonify({'finished':True})

    ef = float(row.ef)
    if ef < 5:
        clz, ans = make_cloze(row.sentence, row.word)
        jp = translate_en_to_jp(row.sentence)  # ★ 追加
        payload = {
            'mode': 'cloze',
            'sentence': clz,
            'full_sentence': row.sentence,
            'answer': ans,
            'jp': jp,  # ★ 追加
            'vocab_id': row.id,
            'ef': ef
        }
    else:
        jp = translate_word_to_jp(row.word)
        payload = {
            'mode'     :'jp',
            'jp'       : jp,
            'word'     : row.word,
            'vocab_id' : row.id,
            'ef'       : ef,
            'full_sentence': row.sentence
        }
    return jsonify(payload)



@app.route('/api/review/result', methods=['POST'])
def post_review_result():
    uid       = session.get('user_id')
    vocab_id  = int(request.form['vocab_id'])
    is_correct= request.form.get('correct') == '1'
    score     = 5 if is_correct else 1

    # 直近データを取得（EF,回数,経過日数 計算）
    prev = get_latest_review(uid, vocab_id)   # helper
    new_ef = calculate_ef_with_decay(
        ef_old    = prev.ef if prev else 2.5,
        self_score= score,
        repetitions= (prev.reps+1) if prev else 1,
        days_since = prev.days_since if prev else 0
    )

    next_on = datetime.utcnow() + timedelta(days=new_ef)
    sql = """INSERT INTO dbo.vocab_reviews
               (user_id,vocab_id,review_time,self_score,ef,next_review)
             VALUES (?,?,?,?,?,?)"""
    execute(sql, uid, vocab_id, datetime.utcnow(), score, new_ef, next_on)
    return jsonify({'ef': round(new_ef,2)})


# app.py（または app_s.py など）に追加
@app.route('/review')
def review_page():
    """
    復習ページ（review.html）を返すだけの GET ルート。
    ユーザー未ログインなら /login へリダイレクト。
    """
    if 'user_id' not in session:
        return redirect(url_for('login_user'))   # 既存のログイン関数名

    return render_template('review.html')


@app.route('/api/tts_b64', methods=['POST'])
def api_tts_b64():
    try:
        data = request.get_json(force=True) or {}
        text  = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'text is required'}), 400

        voice = data.get('voice',  DEFAULT_VOICE)
        style = data.get('style',  DEFAULT_STYLE)

        audio_b64 = tts_to_b64(text, voice=voice, style=style)
        return jsonify({'audio_b64': audio_b64})

    except Exception as e:
        app.logger.exception("TTS error: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route("/test_recorder")
def test_recorder():
    # templates/test_recorder.html を返す
    return render_template("test_recorder.html")


if __name__ == '__main__':
    app.run(debug=True)

