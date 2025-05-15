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
# ファイル冒頭でまとめて
import uuid, re, json, time, random


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

# --------------------- ③ o4-mini 用（STT と同じリソース） ------------------
o4_client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_STT_ENDPOINT"),  # 例: https://xxx.cognitiveservices.azure.com/
    api_key        = os.getenv("AZURE_OPENAI_STT_KEY"),
    api_version    = "2024-12-01-preview"
)
O4_DEPLOY = "o4-mini"          # デプロイ名

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

def generate_example_sentence(idiom: str) -> str:
    uid = session.get('user_id', 0)
    prompt = f"Provide ONE sentence that naturally uses the idiom \"{idiom}\"."

    messages = [
        {"role": "system",
         "content": "You are an assistant that creates natural example sentences."},
        {"role": "user", "content": prompt}
    ]

    rsp = safe_chat(
        user_id    = uid,
        client     = chat_client,
        deployment = CHAT_DEPLOY,
        messages   = messages,
        max_tokens = 60
    )
    return (rsp.choices[0].message.content or "").strip()

def generate_idiom_meaning(idiom: str) -> str:
    uid = session.get('user_id', 0)
    prompt = (
        f'Explain the meaning of the idiom "{idiom}" in simple English, '
        'then give a concise Japanese translation. '
        'Return both in ONE paragraph.'
    )

    messages = [
        {"role": "system",
         "content": "You are an assistant that explains idioms bilingually."},
        {"role": "user", "content": prompt}
    ]

    rsp = safe_chat(
        user_id    = uid,
        client     = chat_client,
        deployment = CHAT_DEPLOY,
        messages   = messages,
        max_tokens = 120
    )
    return (rsp.choices[0].message.content or "").strip()


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
    rate:  str | None = None,   # ← ★ 追加（例 "85%"  "110%"  "-10%" など）
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
    if rate:
        # prosody で速度調整
        text_part = f'<prosody rate="{rate}">{text}</prosody>'
    else:
        text_part = text

    ssml = f"""
<speak version="1.0"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="{voice.split('-')[0]}">
  <voice name="{voice}">
    <mstts:express-as style="{style}">
      {text_part}
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


def translate_en_to_jp(text: str) -> str:
    """
    Azure OpenAI で英文を自然な日本語に翻訳して返す。
    message.content が無い／content-filter で弾かれた場合は空文字列を返す。
    """
    payload = {
        "messages": [
            {"role": "system",
             "content": "You are a professional translator. "
                        "Translate the sentence into natural Japanese."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 120,
        "temperature": 0.0
    }

    try:
        res = requests.post(CHAT_URL, headers=HEADERS,
                            json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()

        # ---------- safe lookup ----------
        content = (
            data.get("choices", [{}])[0]
                .get("message", {})        # regular response
                .get("content")
            or data.get("choices", [{}])[0]
                .get("delta", {})          # streaming-style chunk
                .get("content")
        )

        if content:
            return content.strip()

        app.logger.warning(
            "translate_en_to_jp: no content (finish_reason=%s)",
            data.get("choices", [{}])[0].get("finish_reason")
        )
        return ""          # fall back with empty string

    except Exception as e:
        app.logger.exception("translate_en_to_jp failed: %s", e)
        return ""


def translate_word_to_jp(word: str) -> str:
    """
    単語１語を “日本語３〜５語程度＋頭文字” で返す。
    OpenAI から適切に取れなかった場合は英単語をそのまま返す。
    """
    prompt = (
        f'次の英単語を日本語に翻訳して。およそ３〜５語で。'
        f'さらに、この英単語の最初のアルファベットを書いてください。\n'
        f'Word: "{word}"'
    )

    messages = [
        {"role": "system", "content":
            "You are a bilingual assistant. "
            "Output ONLY the Japanese translation."},
        {"role": "user",   "content": prompt}
    ]

    try:
        # ---------- safe_chat で呼び出し ---------------------------------
        rsp = safe_chat(
            user_id    = session.get('user_id', 0),  # 未ログインなら 0
            client     = chat_client,                # gpt‑4o‑mini 側
            deployment = CHAT_DEPLOY,                # 例: 'tda_4'
            messages   = messages,
            max_tokens = 30,                         # 返答は短い 1 行
            temperature= 0.3
        )

        jp = (rsp.choices[0].message.content or "").strip()
        if not jp:
            raise ValueError("No content in choices")

        return jp.split('\n')[0]       # 複数行返る場合は先頭行だけ

    except RuntimeError as e:          # 日次クォータ超過など
        app.logger.warning("translate_word_to_jp quota: %s", e)
        return word                    # フォールバック

    except Exception as e:
        app.logger.warning("translate_word_to_jp failed: %s", e)
        return word                    # フォールバック



# Azure OpenAI 定数は既存のものを再利用
CHAT_URL = (
    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
    f"{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15"
)
HEADERS  = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

def ai_pick_token(sentence: str, lemma: str) -> str | None:
    uid = session.get('user_id', 0)
    prompt = (
        "From the sentence below, return EXACTLY the token or phrase "
        "that corresponds to the given lemma.\n"
        "- If the lemma is part of a multi-word phrasal verb or idiom, "
        "return the whole phrase as it appears in the sentence.\n"
        "- Output JUST that phrase.\n\n"
        f"Sentence: {sentence}\nLemma: {lemma}"
    )

    messages = [
        {"role": "system",
         "content": "You are a precise linguistic extractor."},
        {"role": "user", "content": prompt}
    ]

    try:
        rsp = safe_chat(
            user_id    = uid,
            client     = chat_client,
            deployment = CHAT_DEPLOY,
            messages   = messages,
            max_tokens = 20
        )
        phrase = (rsp.choices[0].message.content or "").strip()
        if "\n" in phrase or phrase.lower() not in sentence.lower():
            return None
        return phrase
    except Exception:
        app.logger.info("ai_pick_token error")
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


DAILY_LIMITS = {
    'openai_tokens':  100_000,
    'speech_chars':   50_000,
    "vsize_tokens" :  10_000
}

def _today():
    return datetime.now(timezone.utc).date()


def quota_ok(user_id: int, resource: str, delta: int) -> bool:
    """
    delta だけ加算したとき DAILY_LIMITS を超えなければ True
    """
    row = fetchone(
        "SELECT used FROM dbo.api_quota "
        "WHERE user_id=? AND day_utc=? AND resource=?",
        user_id, _today(), resource
    )
    used = row.used if row else 0
    return used + delta <= DAILY_LIMITS.get(resource, float("inf"))

def add_usage(user_id: int, resource: str, delta: int) -> None:
    """
    MERGE で used をインクリメント（INSERT or UPDATE）
    """
    sql = """
    MERGE dbo.api_quota AS T
    USING (SELECT ? AS uid, ? AS dy, ? AS rc) S
      ON (T.user_id=S.uid AND T.day_utc=S.dy AND T.resource=S.rc)
    WHEN MATCHED THEN UPDATE SET used = T.used + ?
    WHEN NOT MATCHED THEN
         INSERT (user_id, day_utc, resource, used)
         VALUES (S.uid, S.dy, S.rc, ?);
    """
    execute(sql, user_id, _today(), resource, delta, delta)


# ------------------------- Chat ラッパー -----------------------
def safe_chat(
    user_id: int,
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict],
    *,
    purpose: str = "openai_tokens",  # ① 資源名を可変に
    **kw
):
    """
    Azure OpenAI Chat 呼び出しを日次トークン制限付きで実行。
    - GPT-4(o) 系: max_tokens
    - o4-mini    : max_completion_tokens
    いずれも推定に含める。
    """
    # ── 1. トークン概算 ──────────────────────────
    est_prompt = sum(len(m.get("content", "")) // 4 for m in messages)
    est_resp   = (
        kw.get("max_tokens") or
        kw.get("max_completion_tokens") or
        0
    )
    delta = est_prompt + est_resp

    if not quota_ok(user_id, purpose, delta):
        raise RuntimeError("日次トークン上限を超えました。翌日までお待ち下さい。")

    # ── 2. 実リクエスト ──────────────────────────
    rsp = client.chat.completions.create(
        model    = deployment,
        messages = messages,
        **kw
    )

    # ── 3. 実使用トークンで集計 ──────────────────
    real_used = rsp.usage.prompt_tokens + rsp.usage.completion_tokens
    add_usage(user_id, purpose, real_used)

    return rsp

# ------------------------- Whisper 秒数 -----------------------
def safe_stt(user_id: int, audio_sec: int):
    """
    Whisper v2 の音声長(秒)をクォータ管理
    """
    if "whisper_sec" not in DAILY_LIMITS:        # 制限しない設定
        return
    if not quota_ok(user_id, "whisper_sec", audio_sec):
        raise RuntimeError("Daily STT quota exceeded 🎤")
    add_usage(user_id, "whisper_sec", audio_sec)


@app.route('/login', methods=['GET', 'POST'])
def login_user():
    # --------------- GET: 画面表示 -----------------
    if request.method == 'GET':
        return render_template('login.html')

    user_id_input  = request.form.get('user_id', '').strip()
    password_input = request.form.get('password')

    # --------------- ① 認証チェック -----------------
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, password_hash, name, is_email_verified
          FROM dbo.users
         WHERE user_id = ?
    """, user_id_input)
    row = cur.fetchone()
    cur.close(); conn.close()      # ← ★ ここで **必ず**閉じてしまう

    if not row:
        return render_template('login.html', message='ユーザーが存在しません')

    db_id, db_pw_hash, db_name, verified = row
    if not verified:
        return render_template('login.html', message='メール認証を完了してください')

    if not check_password(password_input, db_pw_hash):
        return render_template('login.html', message='パスワードが違います')

    # --------------- ② セッション保存 ---------------
    session['user_id']   = db_id
    session['user_name'] = db_name

    # --------------- ③ パーソナルコース存在確認 -----
    conn = get_db_connection()     # ★ 新しい接続を取得
    cur  = conn.cursor()
    cur.execute("""
        SELECT 1
          FROM dbo.courses
         WHERE owner_user_id = ? AND category_id = 4
    """, db_id)
    has_personal = cur.fetchone() is not None
    cur.close(); conn.close()      # 使い切ったら必ずクローズ

    # --------------- ④ リダイレクト ----------------
    if has_personal:
        return redirect(url_for('home'))
    else:
        return redirect(url_for('level_select'))


# app_s.py  ─ home() ルート
@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))

    uid = session['user_id']

    # ⬇︎ ① すでにある helper を再利用して “個人コース名” を取得
    _, personal_name = ensure_personal_course(uid)   # returns (id, name)

    # ⬇︎ ② personal_course をテンプレへ渡す
    return render_template(
        'home.html',
        user_name      = session.get('user_name'),
        user_id        = uid,
        personal_course= personal_name              # ★ 追加
    )


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

# @app.route('/encourage', methods=['POST'])
# def encourage():
#     try:
#         language = random.choice(['en', 'zh'])
#         message = generate_encouraging_message(language)
#         access_token = get_access_token()
#
#         if language == 'en':
#             language_code = 'en-US'
#             voice_name = 'en-US-AriaNeural'
#         else:
#             language_code = 'zh-CN'
#             voice_name = 'zh-CN-XiaoxiaoNeural'
#
#         audio_content = generate_speech(message, access_token, language_code, voice_name, style='whispering')
#         audio_base64 = base64.b64encode(audio_content).decode('utf-8')
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#     return jsonify({'message': message, 'audio': audio_base64})

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

def build_ws_audio(items):
    combo = AudioSegment.silent(duration=0)
    for it in items:
        text = f"{it['word']}. {it['sentence']}"
        for _ in range(3):                               # 3 回
            bin_mp3 = generate_speech(
                text, get_access_token(),
                'en-US', 'en-US-GuyNeural')
            seg = AudioSegment.from_file(BytesIO(bin_mp3), format='mp3')
            combo += seg + AudioSegment.silent(duration=400)
    buf = BytesIO(); combo.export(buf, format='mp3')
    return base64.b64encode(buf.getvalue()).decode('ascii')


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


# ────────────────────────────────────────────────────────────
#  /api/get_due_vocab  ―  “在庫切れなら低-EF を再抽出” 版
# ────────────────────────────────────────────────────────────
# ① 10 分クールダウンを尊重した通常クエリ
SQL_DUE = """
    SELECT
        vi.id        AS vocab_id,
        vi.sentence,
        vi.word,
        ISNULL(lat.ef, 2.5) AS ef
    FROM dbo.vocab_items vi
    LEFT JOIN (
        SELECT vocab_id, ef, review_time,
               ROW_NUMBER() OVER (PARTITION BY vocab_id ORDER BY review_time DESC) rn
        FROM dbo.vocab_reviews
        WHERE user_id = ?
    ) lat ON vi.id = lat.vocab_id AND lat.rn = 1
    WHERE vi.course = ?
      AND (
            lat.review_time IS NULL
         OR lat.review_time <= DATEADD(minute,-10,GETUTCDATE())
      )
    ORDER BY
        ISNULL(lat.ef, 3.0) ASC,
        ISNULL(lat.review_time,'1900-01-01') ASC,
        NEWID()
"""

# ② フォールバック：クールダウン無視・低 EF 優先（最大 20 語）
SQL_FALLBACK = """
    SELECT TOP 20
        vi.id        AS vocab_id,
        vi.sentence,
        vi.word,
        ISNULL(lat.ef, 2.5) AS ef
    FROM dbo.vocab_items vi
    LEFT JOIN (
        SELECT vocab_id, ef,
               ROW_NUMBER() OVER (PARTITION BY vocab_id ORDER BY review_time DESC) rn
        FROM dbo.vocab_reviews
        WHERE user_id = ?
    ) lat ON vi.id = lat.vocab_id AND lat.rn = 1
    WHERE vi.course = ?
    ORDER BY ISNULL(lat.ef, 2.5) ASC, NEWID()
"""

@app.route('/api/get_due_vocab')
def get_due_vocab():
    """Shadowing 用: due が無ければクールダウン無視で再抽出"""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'ログインしてください'}), 403

    course = request.args.get('course', '')
    if not course:
        return jsonify({'error': 'course パラメータが必要です'}), 400

    conn = get_db_connection(); cur = conn.cursor()

    # ---------- ① 通常の due 抽出 ----------
    cur.execute(SQL_DUE, uid, course)
    rows = cur.fetchall()

    # ---------- ② 0 行ならフォールバック ----------
    if not rows:
        cur.execute(SQL_FALLBACK, uid, course)
        rows = cur.fetchall()

    cur.close(); conn.close()

    # ---------- 整形して返却 ----------
    return jsonify([{
        'vocab_id': r.vocab_id,
        'sentence': r.sentence,
        'word':     r.word,
        'ef':       float(r.ef)
    } for r in rows])


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

    try:
        chat = safe_chat(
            user_id    = session.get('user_id', 0),  # 未ログインなら 0
            client     = chat_client,                # gpt-4o-mini 用クライアント
            deployment = CHAT_DEPLOY,                # 例: "tda_4"
            messages   = messages,
            max_tokens = 1000,
            temperature= 0.7,
            response_format={"type": "json_object"}
        )
    except RuntimeError as e:                       # 日次トークン上限超過など
        return jsonify({"error": str(e)}), 429

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


# ① 英語っぽいトークンに単純マッチ（A-Z だけで出来ている語を英語とみなす）
ENG_WORD = re.compile(r'[A-Za-z]{2,}')

def text_to_ssml(text: str,
                 jp_voice="ja-JP-NanamiNeural",
                 en_voice="en-US-AriaNeural",
                 jp_style="cheerful",
                 en_style="chat") -> str:
    """
    文章中の「英単語／英文らしき部分」を自動判定し、
    日本語は jp_voice、英語は en_voice で読み上げる SSML を返す。
    """
    # ① 英語らしいブロックを抽出  ─ 連続する ASCII 文字列
    tokens = re.split(r'([A-Za-z0-9 ,.;:!?\'"()-]+)', text)

    ssml_parts = []
    for tok in tokens:
        if not tok:
            continue
        # “英語ブロック” とみなす簡易判定（ASCII 率 80% 以上）
        ascii_ratio = sum(1 for c in tok if ord(c) < 128) / len(tok)
        is_en = ascii_ratio > 0.8

        if is_en:
            ssml_parts.append(
                f'<voice name="{en_voice}">'
                f'  <mstts:express-as style="{en_style}">{tok}</mstts:express-as>'
                f'</voice>')
        else:
            ssml_parts.append(
                f'<voice name="{jp_voice}">'
                f'  <mstts:express-as style="{jp_style}">{tok}</mstts:express-as>'
                f'</voice>')

    return '''
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="ja-JP">
  {}
</speak>'''.format('\n  '.join(ssml_parts)).strip()


def tts_ssml(access_token: str, ssml: str) -> bytes:
    """
    SSML をそのまま Azure TTS に投げる超シンプル版。
    voice/style を含む SSML を自前で組み立ててから渡す時専用。
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/ssml+xml; charset=utf-8",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
    }
    url = os.getenv("AZURE_TTS_API_URL")
    r = requests.post(url, headers=headers, data=ssml.encode("utf-8"))
    r.raise_for_status()
    return r.content


def sakura_teacher_inner(sentence: str, words: list[str]) -> dict:
    word_list = ', '.join(f'「{w}」' for w in words)
    sys_prompt = (
        "あなたは『さくら先生』というやさしい日本語教師です。\n"
        f"Sentence: 「{sentence}」\n"
        f"Words: {word_list}\n\n"
        "1)まずSentence について意味を簡単に解説し、\n"
        "2)Words について一つずつ意味を解説してください\n"
        "Wordsについて、カタカナを使って解説しないでください\n"
    )
    gpt = safe_chat(
        user_id=session.get('user_id', 0),
        client=chat_client,
        deployment=CHAT_DEPLOY,  # "tda_4"
        messages=[{"role": "system", "content": sys_prompt}],
        max_tokens=400,
        temperature=0.7
    )
    jp_text = gpt.choices[0].message.content.strip()

    ssml     = text_to_ssml(jp_text)                       # JP=Nanami / EN=Aria
    audio_b64= base64.b64encode(
                 tts_ssml(get_access_token(), ssml)
               ).decode('ascii')
    return {"text": jp_text, "audio": audio_b64}


@app.route('/api/sakura', methods=['POST'])
def sakura_teacher():
    """
    さくら先生:
      1. 単語をやさしく解説
      2. 例文の意味
      3. Collocations
      4. 励ましメッセージ
    日本語は Nanami (ja-JP)、英語は Aria (en-US) で読み上げる。
    """
    data = request.get_json(force=True)
    word = data.get('word', '').strip()
    sentence = data.get('sentence', '').strip()

    # ---------- ① GPT で日本語説明を生成 ----------
    prompt = f"""
あなたは『さくら先生』というやさしい日本語教師です。
Word: 「{word}」
Sentence: 「{sentence}」

1. 単語を小学生にも分かるように簡単に解説
2. 例文の意味を解説
3. 単語について頻繁に使われる Collocations コロケーションをいくつか紹介
4. 最後に相手を励ます短いメッセージ
口調は親しみやすく、語尾に♡などは使わず自然体で。
"""
    gpt = chat_client.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
    )
    jp_text = gpt.choices[0].message.content.strip()

    # ---------- ② 日本語＋英語混在 SSML を生成 ----------
    ssml = text_to_ssml(jp_text)

    # ---------- ③ Azure Neural TTS で音声化 ----------
    try:
        access = get_access_token()
        audio_bin = tts_ssml(access, ssml)
        audio_b64 = base64.b64encode(audio_bin).decode('utf-8')
    except Exception as e:
        app.logger.exception("TTS failed: %s", e)
        # 音声が生成できなくてもテキストだけは返す
        audio_b64 = ""

    return jsonify({'text': jp_text, 'audio': audio_b64})


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

    # ――― ① EF < 5 → cloze ―――
    if ef < 5:
        clz, ans = make_cloze(row.sentence, row.word)
        jp = translate_en_to_jp(row.sentence)
        return jsonify({
            'mode': 'cloze',
            'sentence': clz,
            'full_sentence': row.sentence,
            'answer': ans,
            'jp': jp,
            'vocab_id': row.id,
            'ef': ef
        })

    # ――― ② 5 ≦ EF < 6 → JP ➜ EN 変換 ―――
    if ef < 6:
        jp = translate_word_to_jp(row.word)  # ← 1行日本語
        return jsonify({
            'mode': 'jp',
            'jp': jp,
            'word': row.word,
            'full_sentence': row.sentence,
            'vocab_id': row.id,
            'ef': ef
        })

    # ――― ③ EF ≥ 6 → ディクテーション ―――
    # （sentence を全文 TTS して base64 で渡す）
    audio_b64 = tts_to_b64(row.sentence)
    return jsonify({
        'mode': 'dict',
        'audio': audio_b64,  # ★ text は送らない
        'full_sentence': row.sentence,  # 判定用
        'vocab_id': row.id,
        'ef': ef
    })


@app.route('/api/review/result', methods=['POST'])
def post_review_result():
    uid       = session.get('user_id')
    vocab_id  = int(request.form['vocab_id'])
    score    = int(request.form['score'])

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


@app.route('/multiplication')
def multiplication():
    return render_template('multiplication.html')


@app.route('/api/multiplication_log', methods=['POST'])
def multiplication_log():
    data = request.get_json(force=True) or {}
    expr    = data.get('expression', '').strip()
    answer  = data.get('answer')
    correct = bool(data.get('correct'))
    uid     = session.get('user_id')          # 未ログインなら None

    if not expr or answer is None:
        return jsonify({'error': 'bad request'}), 400

    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO dbo.multip_logs (user_id, expression, answer, is_correct)
            VALUES (?, ?, ?, ?)
        """, uid, expr, int(answer), 1 if correct else 0)
        conn.commit()
        cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        app.logger.exception('multiplication_log insert failed: %s', e)
        return jsonify({'error': 'db error'}), 500


@app.route('/multiplication_dashboard')
def multiplication_dashboard():
    return render_template('multiplication_dashboard.html')


@app.route('/api/multiplication_stats')
def multiplication_stats():
    """
    Return aggregated stats for the multiplication app.
    Response JSON structure:
      {
        "stats": [
          {"expression": "3*4", "attempts": 5, "correct": 4},
          ...
        ]
      }
    """
    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        cur.execute("""
            SELECT
                expression,
                COUNT(*)                                   AS attempts,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct
            FROM dbo.multip_logs
            GROUP BY expression
        """)

        rows = [
            {
                "expression": r.expression,
                "attempts":   int(r.attempts),
                "correct":    int(r.correct)
            }
            for r in cur
        ]
        return jsonify({"stats": rows})

    except Exception as e:
        app.logger.exception("multiplication_stats failed: %s", e)
        return jsonify({"error": "db error"}), 500

    finally:
        if cur:  cur.close()
        if conn: conn.close()


@app.route('/api/get_multip_question')
def get_multip_question():
    base = request.args.get('base', type=int, default=0)  # 0 = random
    if base and (base < 2 or base > 9):
        return jsonify({'error': 'invalid base'}), 400

    import random
    if base:
        a = base
        b = random.randint(2, 9)
    else:
        a = random.randint(2, 9)
        b = random.randint(2, 9)

    # a と b をランダムに入れ替えても OK
    if random.random() < 0.5:
        a, b = b, a

    return jsonify({'a': a, 'b': b})


# ───────────────────────────
#  Vocabulary Size Quick-Test
# ───────────────────────────
# --------------------------------------------
# 50 語語彙テスト初期化  ― シンプル版
# --------------------------------------------
import uuid, random, json
# ----------------- ② サーバ側バックアップ語列（実在語 40） -------
BACKUP_REAL = [
    "angle","bicycle","cabinet","diminish","fragile","horizon","imitate",
    "jovial","kernel","linger","mundane","nebulous","opaque","parity",
    "quiver","raucous","salient","tenuous","unfurl","verbose","wistful",
    "xenophobia","yearling","zealous","abrogate","bellicose","cabal",
    "diaphanous","enervate","fractious","galvanize","hauteur",
    "impecunious","jejune","lachrymose","mendacity","nonplussed",
    "obdurate","perspicacious","quotidian","respite","sagacious"
][:40]                                   # 念のため 40 語に切り詰め

# ----------------- ③ GPT に 10 語生成させるヘルパ ---------------
def gpt_ten_words() -> list[dict]:
    """
    REAL 5 ＋ FAKE 5 を o4-mini で生成して返す。
    返り値例:
      [{'word': 'angle', 'fake': False}, …]  ← 必ず 10 要素
    """
    prompt = (
        "Return exactly 10 items as ONE JSON array. "
        "First 5 are REAL English lemmas (\"fake\":false). "
        "Next 5 are plausible pseudo-words (\"fake\":true). "
        "Format: [{\"word\":\"angle\",\"fake\":false}, … ]"
    )

    # ── ★ ここだけ safe_chat で呼び出す ─────────────────
    try:
        rsp = safe_chat(
            user_id    = session.get('user_id', 0),   # 未ログインなら 0
            client     = o4_client,                   # o4-mini 用クライアント
            deployment = O4_DEPLOY,                   # "o4-mini"
            messages   = [{"role": "user", "content": prompt}],
            max_completion_tokens = 3000
        )
        raw = (rsp.choices[0].message.content or "").strip()
    except RuntimeError as e:                         # 日次上限など
        app.logger.warning("gpt_ten_words quota: %s", e)
        return []                                    # 呼び元で再試行

    # ── JSON パース & 最低限のバリデーション ────────────
    try:
        obj   = json.loads(raw)
        items = obj["words"] if isinstance(obj, dict) else obj
    except Exception:
        app.logger.warning("gpt_ten_words JSON parse fail: %s", raw[:120])
        return []                                    # 呼び元で再試行

    out, seen = [], set()
    for it in items:
        w = str(it.get("word", "")).strip()
        f = bool(it.get("fake", False))
        if w and w.lower() not in seen:
            seen.add(w.lower())
            out.append({"word": w, "fake": f})
        if len(out) == 10:
            break

    return out if len(out) == 10 else []

# ----------------- ④ Flask ルート -------------------------------
@app.route("/api/vsize/start")
def vsize_start():
    if "user_id" not in session:
        return jsonify({"error":"login"}),403

    # 1) GPT で 10 語取得（最大 3 回リトライ）
    gpt_items, tries = [], 0
    while len(gpt_items) < 10 and tries < 3:
        gpt_items = gpt_ten_words()
        tries += 1
        time.sleep(0.5)

    if len(gpt_items) < 10:
        return jsonify({"error":"gpt failed"}),502

    # 2) バックアップ 40 語と結合して 50 語リストを作成
    words = (
        [{"id":i+1, "word":w, "fake":False}
         for i,w in enumerate(BACKUP_REAL)] +
        [{"id":i+41, "word":it["word"], "fake":it["fake"]}
         for i,it in enumerate(gpt_items)]
    )
    random.shuffle(words)

    # 3) セッション保存
    tid = str(uuid.uuid4())
    session["vsize_test"] = {"id":tid, "words":words}

    # 4) クライアント返送（fake フラグは隠す）
    public = [{"id":w["id"], "word":w["word"]} for w in words]
    return jsonify({"test_id":tid,"items":public})

# ---------------------------------------------------------
# 採点エンドポイント  /api/vsize/submit
# ---------------------------------------------------------
REAL_TOTAL  = 40          # vsize_start で固定
FAKE_TOTAL  = 10
TARGET_REAL_RATIO = 48000 # (= 本テストで 100 % 正答した場合の推定レマ数)

@app.route("/api/vsize/submit", methods=["POST"])
def vsize_submit():
    j = request.get_json(force=True) or {}
    test = session.get("vsize_test")

    # ① セッション検証
    if not test or test["id"] != j.get("test_id"):
        return jsonify({"error": "session expired"}), 400

    chosen = set(int(i) for i in j.get("known_ids", []))

    # ② real / fake を分類
    real = [w for w in test["words"] if not w["fake"]]
    fake = [w for w in test["words"] if w["fake"]]

    real_known = sum(1 for w in real if w["id"] in chosen)
    fake_hit   = sum(1 for w in fake if w["id"] in chosen)

    # ③ 補正 (擬似語誤認を差し引き、下限 0)
    net = max(real_known - fake_hit, 0)

    pct = net / REAL_TOTAL                   # 実在語正答率 (0.0–1.0)

    est_lemmas   = int(round(pct * TARGET_REAL_RATIO, -2))   # 100 語単位
    est_families = int(round(est_lemmas / 2.2, -1))          # 10 語単位

    return jsonify({
        "net":          net,
        "pct":          round(pct * 100, 1),
        "fake_hit":     fake_hit,
        "est_lemmas":   est_lemmas,
        "est_families": est_families
    })


@app.route("/api/vsize/report", methods=["POST"])
def vsize_report():
    """
    受信例:
      {"net":14,"pct":35.0,"est_lemmas":17000,"est_families":8000,"fake_hit":1}
    戻り値:
      {"report":"<日本語レポート全文>"}
    """
    data = request.get_json(force=True) or {}

    uid = session.get('user_id', 0)

    sys_prompt = (
        "あなたは英語教育の専門家です。日本語母語話者の学習者に向けて、"
        "語彙診断の結果を踏まえた詳細な学習アドバイスを作成してください。"
    )

    user_prompt = f"""
【必須セクション】
1. 評価表（項目 / あなたの結果 / 一般的な日本人EFL学習者の目安 / コメント）
2. 推定語彙量、推定TOEICレベル、推定英検レベル、推定CEFRレベル
3. 強みと改善点
4. 結論

【条件】
- 得点や語彙量が低い場合は初級〜中級向け、高い場合は上級向けとコメントを変えること
- レマ、CEFRなど一般人にはなじみのない概念ついて、簡単に解説してください
- プレーンテキストで出力（Markdown・HTML禁止）
- 下記 JSON を参照して数値を反映すること

### 診断結果 JSON
{json.dumps(data, ensure_ascii=False)}
"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    try:
        rsp = safe_chat(
            user_id=uid,  # ← 呼び出しユーザー
            client=o4_client,  # ← どのリソースか
            deployment=O4_DEPLOY,  # ← デプロイ名
            messages=messages,
            max_completion_tokens=3000,
            purpose="vsize_tokens"
        )
    except RuntimeError as e:  # 日次上限を超えたとき
        return jsonify({"error": str(e)}), 429



    report_text = (rsp.choices[0].message.content or "").strip()
    return jsonify({"report": report_text})


@app.route('/vocab_test')
def vocab_test_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('vocab_test.html')


# ───────────── A) post-login 判定 ─────────────
@app.route('/post_login_redirect')
def post_login_redirect():
    """
    ログイン成功後に呼び出す。
    owner_user_id == 自分 かつ category_id == 4 のコース
    （= パーソナル辞書）が存在すれば /home、
    無ければレベル選択へ。
    """
    if 'user_id' not in session:
        return redirect(url_for('login_user'))

    uid = session['user_id']
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT 1
          FROM dbo.courses
         WHERE owner_user_id = ? AND category_id = 4
    """, uid)
    has_personal = bool(cur.fetchone())
    cur.close(); conn.close()

    return redirect(url_for('home' if has_personal else 'level_select'))


# ----------------  レベル → コース名 マッピング  ----------------
COURSE_MAP = {
    'L1': 'L1_Starter',
    'L2': 'L2_Basic',
    'L3': 'L3_UpperIntermediate',
    'L4': 'L4_Advanced',
    'L5': 'L5_Expert',
    'L6': 'L6_Master'
}

# --------------------------------------------------------------
#  レベル選択画面 ── 10 語取得エンドポイント
#    例: /api/level_words?level=L3
# --------------------------------------------------------------
@app.route('/api/level_words')
def api_level_words():
    """
    ▸ パラメータ:  level=L1〜L6
    ▸ 処理:
        1. level を COURSE_MAP でコース名へ変換
        2. そのコースに登録されている単語を取得（10 語想定）
        3. { items:[{id,word,sentence}, …] } を返す
    """
    # ① 認証
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403

    # ② level → コース名解決
    level = request.args.get('level', '').upper()
    course_name = COURSE_MAP.get(level)
    if not course_name:
        return jsonify({'error': 'level'}), 400

    # ③ DB から語彙取得
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, word, sentence
          FROM dbo.vocab_items
         WHERE course = ?
         ORDER BY id
    """, course_name)
    items = [dict(id=r.id, word=r.word, sentence=r.sentence) for r in cur]
    cur.close(); conn.close()

    # ④ 検証（10 語なければエラー）
    if len(items) != 10:
        return jsonify({'error': f'course "{course_name}" must contain 10 items'}), 500

    return jsonify({'items': items})


def ensure_personal_course(uid:int)->tuple[int,str]:
    """
    personal コースを探し、なければ作成して
    (course_id, course_name) を返す
    """
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT id, name
          FROM dbo.courses
         WHERE owner_user_id=? AND category_id=4
    """, uid)
    row = cur.fetchone()

    if row:                         # 既存
        cid, cname = row.id, row.name
    else:                           # 新規作成
        cname = f"{session['user_name']}_personal"   # ←ご要望どおり
        cur.execute("""
            INSERT INTO dbo.courses
                   (name, language, is_public, owner_user_id, category_id, overview)
            OUTPUT INSERTED.id
            VALUES (?, 'en', 0, ?, 4, N'あなただけのパーソナライズ辞書')
        """, cname, uid)
        cid = cur.fetchone()[0]
        conn.commit()

    cur.close(); conn.close()
    return cid, cname


@app.route('/api/level_confirm', methods=['POST'])
def api_level_confirm():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403

    data  = request.get_json(force=True) or {}
    items = data.get('items', [])
    if len(items) != 10:
        return jsonify({'error': 'need 10 items'}), 400

    uid = session['user_id']
    course_id, course_name = ensure_personal_course(uid)

    conn = get_db_connection(); cur = conn.cursor()
    for it in items:
        cur.execute("""
            INSERT INTO dbo.vocab_items (course_id, course, sentence, word)
            VALUES (?, ?, ?, ?)
        """, course_id, course_name, it['sentence'], it['word'])
    conn.commit()
    cur.close(); conn.close()

    return jsonify({'ok': True})


@app.route('/level_select')
def level_select():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('level_select.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_pw.html')

    email = request.form.get('email', '').strip().lower()
    if not email:
        return render_template('forgot_pw.html', msg='メールを入力してください')

    conn, cur = get_db_connection(), None
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM dbo.users WHERE email = ?", email)
        row = cur.fetchone()
        if not row:
            return render_template('forgot_pw.html', msg='登録されていません')

        uid, name = row.id, row.name
        token = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + timedelta(minutes=30)

        cur.execute("""
            UPDATE dbo.users
               SET reset_token = ?, reset_expiry = ?
             WHERE id = ?
        """, token, expiry, uid)
        conn.commit()
    finally:
        if cur: cur.close()
        conn.close()

    # ── メール送信 ──────────────────────────────
    reset_link = f"{VERIFY_BASE_URL.replace('verify_email','reset_password')}?token={token}"
    body = f"""{name} さん

以下のリンクをクリックして新しいパスワードを設定してください。
30 分以内に完了しないと無効になります。

{reset_link}

Polyagent AI"""
    mail.send(Message(
        subject='[Polyagent AI] パスワード リセット',
        recipients=[email],
        body=body
    ))
    return render_template('forgot_pw_done.html')

# ──────────────────────────────────────────────
# ② リセットリンク → 新パスワード入力
# ──────────────────────────────────────────────
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    token = request.args.get('token','').strip() if request.method == 'GET' else request.form.get('token','')
    if not token:
        return render_template('reset_pw_result.html', success=False, msg='リンクが無効です')

    conn, cur = get_db_connection(), None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, reset_expiry FROM dbo.users
             WHERE reset_token = ?
        """, token)
        row = cur.fetchone()
        if not row or row.reset_expiry < datetime.utcnow():
            return render_template('reset_pw_result.html', success=False, msg='リンクが無効または期限切れです')

        uid = row.id

        # GET → フォーム表示
        if request.method == 'GET':
            return render_template('reset_pw.html', token=token)

        # POST → パスワード更新
        pw1 = request.form.get('password','')
        pw2 = request.form.get('password2','')
        if len(pw1) < 6 or pw1 != pw2:
            return render_template('reset_pw.html', token=token, msg='確認用が一致しません')

        cur.execute("""
            UPDATE dbo.users
               SET password_hash = ?, reset_token = NULL, reset_expiry = NULL
             WHERE id = ?
        """, hash_password(pw1), uid)
        conn.commit()
        return render_template('reset_pw_result.html', success=True, msg='パスワードを更新しました')
    finally:
        if cur: cur.close()
        conn.close()


@app.route('/courses')
def courses_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))      # 未ログインならログイン画面へ

    return render_template(
        'courses.html',
        user_id   = session['user_id'],             # 使わなければ省略可
        user_name = session.get('user_name', '')    # 〃
    )


# ─────────────────────────────────────────────────────────
#  /api/train_round
#    1. パーソナル辞書から「EF が低い or 未学習」の語を 5 語取得
#    2. 単語+例文を 3 回読む音声 (intro / review) を生成
#    3. 5 語すべてを含む英文を GPT で生成し
#       既存関数 sakura_teacher() で日本語解説+音声を取得
#    4. 3 種の音声 + テキストを JSON で返す
# ─────────────────────────────────────────────────────────
from pydub import AudioSegment
from io    import BytesIO
import base64

@app.route('/api/train_round')
def api_train_round():
    # ── ① 認証とパーソナル辞書名 ─────────────────────
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'login'}), 403

    _, personal_name = ensure_personal_course(uid)   # category_id = 4

    # ── ② EF が低い or 未学習の語を 5 語抽出 ─────────
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        WITH latest AS (
            SELECT vocab_id, ef,
                   ROW_NUMBER()OVER(PARTITION BY vocab_id
                                     ORDER BY review_time DESC) rn
              FROM dbo.vocab_reviews
             WHERE user_id = ?
        )
        SELECT TOP 5
               vi.id, vi.word, vi.sentence,
               ISNULL(lat.ef, 0) AS ef
          FROM dbo.vocab_items vi
          LEFT JOIN latest lat
                 ON vi.id = lat.vocab_id AND lat.rn = 1
         WHERE vi.course = ?
           AND NOT EXISTS (
                SELECT 1
                  FROM dbo.train_round_seen s
                 WHERE s.user_id   = ?
                   AND s.vocab_id  = vi.id
                   AND s.seen_time >= DATEADD(minute, -10, GETUTCDATE())
           )
         ORDER BY ISNULL(lat.ef,0) ASC,
                  NEWID()
    """, uid, personal_name, uid)
    rows = cur.fetchall()

    if len(rows) < 5:  # パーソナル辞書が足りない
        short = 5 - len(rows)  # まだ必要な語数

        cur.execute("""
                    /*―― ① ユーザーの最新レビュー行を作る ――*/
                    WITH latest AS (SELECT vocab_id,
                                           ef,
                                           ROW_NUMBER() OVER (
                           PARTITION BY vocab_id
                           ORDER BY review_time DESC) AS rn
                                    FROM dbo.vocab_reviews
                                    WHERE user_id = ?)
                    /*―― ② EF が低い順に不足ぶんだけ取得 ――*/
                    SELECT TOP(?) vi.id, vi.word,
                           vi.sentence,
                           ISNULL(lat.ef, 0) AS ef
                    FROM latest AS lat
                             JOIN dbo.vocab_items AS vi
                                  ON vi.id = lat.vocab_id
                    WHERE lat.rn = 1 -- vocab_id ごと最新 1 行
                      AND lat.ef < 10 -- ★低 EF だけ
                      AND NOT EXISTS ( -- ★10 分以内に出題していない
                        SELECT 1
                        FROM dbo.train_round_seen AS s
                        WHERE s.user_id = ?
                          AND s.vocab_id = vi.id
                          AND s.seen_time >= DATEADD(minute,-10, GETUTCDATE()))
                    ORDER BY lat.ef ASC, NEWID(); -- ★EF が小さいほど先に
                    """, (uid, short, uid))

        rows.extend(cur.fetchall())

    # ── ③-B 5 語未満ならランダム補充 ──────────────
    if len(rows) < 5:
        cur.execute("""
            SELECT TOP 5 id, word, sentence
              FROM dbo.vocab_items
             WHERE course = ?
             ORDER BY NEWID()
        """, personal_name)
        rows = cur.fetchall()

    words = [{"id": r.id, "word": r.word, "sentence": r.sentence} for r in rows]

    if len(words) < 5:
        cur.close(); conn.close()
        return jsonify({"error": "need ≥5 items"}), 500

    # ── ③-C 取得した語を履歴テーブルに登録 ──────
    cur.executemany(
        "INSERT INTO dbo.train_round_seen(user_id, vocab_id) VALUES (?,?)",
        [(uid, w["id"]) for w in words]
    )
    conn.commit()
    cur.close(); conn.close()

    intro_b64  = build_ws_audio(words)   # 最初
    review_b64 = build_ws_audio(words)   # おさらい

    # ── ④ 5 語入り英文を GPT で生成 ─────────────────
    prompt = ("Teach me most common collocation of one of the words: "
              + ', '.join(w['word'] for w in words) + '.')
    gpt = chat_client.chat.completions.create(
        model=CHAT_DEPLOY,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=80,
        temperature=0.7
    )
    example_sentence = gpt.choices[0].message.content.strip()

    example_audio_b64 = tts_to_b64(
        example_sentence,
        voice="en-US-OnyxTurboMultilingualNeural",
        style="general",
        rate="0.9"
    )

    # ── ⑤ さくら先生で解説と音声 ────────────────────

    sakura_res = sakura_teacher_inner(
        sentence=example_sentence,
        words=[w['word'] for w in words])  # 5 語全部
    # 期待返り値: {'text': ..., 'audio': ...}

    # ── ⑥ クライアントへ返却 ────────────────────────
    return jsonify({
        'words'       : words,                 # [{word, sentence} ×5]
        'intro_audio' : intro_b64,             # 単語+例文×3 (前半)
        "example_sentence": example_sentence,
        'sentence_audio': example_audio_b64,
        'story_audio' : sakura_res['audio'],   # さくら先生 (例文解説)
        'review_audio': review_b64,            # 単語+例文×3 (後半)
        'sakura_text' : sakura_res['text']     # 日本語解説テキスト
    })


@app.route('/train')
def train_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('train.html',
                           user_id   = session['user_id'],
                           user_name = session.get('user_name',''))


@app.route('/profile')
def vocab_profile_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('profile.html')


@app.route('/api/personal_vocab/profile')
def vocab_profile_api():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403
    uid = session['user_id']

    # ── ① パーソナル辞書 (category_id=4) の全単語を取得 ─────────
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT vi.word
          FROM dbo.courses c
          JOIN dbo.vocab_items vi ON vi.course_id = c.id
         WHERE c.owner_user_id = ? AND c.category_id = 4
           AND vi.word IS NOT NULL
    """, uid)
    words = [r.word for r in cur.fetchall()]
    cur.close(); conn.close()

    if not words:
        return jsonify({'error': 'empty'}), 404

    sample = ', '.join(words[:400])        # 多すぎるとプロンプト肥大
    total  = len(words)

    # ── ② o4-mini にレポートを依頼（JSON で返させる） ────────
    prompt = (
        "You are an English-vocabulary analyst.\n"
        f"The learner has {total} personal dictionary entries. "
        "Here is a sample list:\n"
        f"{sample}\n\n"
        "Estimate:\n"
        "1. Total lemma size (English lemmas they probably know)\n"
        "2. CEFR level (A1–C2)\n"
        "Return ONE JSON object with keys \"lemmas\", \"cefr\", \"comment\".\n"
        "comment MUST be ≤40 English words."
    )

    # ① ユーザー ID を取得
    uid = session.get('user_id', 0)

    # ② safe_chat に置き換え
    try:
        rsp = safe_chat(
            user_id=uid,  # ← クォータを引く対象
            client=o4_client,  # ← どのリソースか
            deployment=O4_DEPLOY,  # ← "o4-mini"
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
    except RuntimeError as e:  # 上限超過など
        return jsonify({"error": str(e)}), 429

    try:
        data = json.loads(rsp.choices[0].message.content)
    except Exception:
        return jsonify({'error': 'parse'}), 502

    # ───────── ★ ここに “users テーブル更新” を追記 ─────────
    conn = get_db_connection();
    cur = conn.cursor()
    cur.execute("""
                UPDATE dbo.users
                SET vocab_lemmas          = ?,
                    vocab_cefr            = ?,
                    vocab_comment         = ?,
                    vocab_profile_updated = GETUTCDATE()
                WHERE id = ?
                """,
                int(data.get('lemmas', 0)),
                str(data.get('cefr', ''))[:2],  # 例: "B2"
                data.get('comment', '')[:200],  # 200 文字に切り詰め
                uid  # ← session の user_id
                )
    conn.commit()
    cur.close();
    conn.close()
    # ────────────────────────────────────────────────

    return jsonify({'ok': True, **data})


@app.route('/api/personal_vocab/profile_cached')
def vocab_profile_cached():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403

    uid = session['user_id']
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT vocab_lemmas,
               vocab_cefr,
               vocab_comment,
               vocab_profile_updated
          FROM dbo.users
         WHERE id = ?
    """, uid)
    row = cur.fetchone()
    cur.close(); conn.close()

    # 1) まだプロフィールが無ければ再計算を促す
    if not row or row.vocab_lemmas is None:
        return jsonify({'ok': False})      # → フロント側が /profile を再呼び出し

    # 2) 「古さ」を 60 日で判定（aware / naive どちらでも動くように調整）
    last = row.vocab_profile_updated
    if last:                               # last が None の可能性もある
        if last.tzinfo is None:            # ← naive なら UTC とみなして tzinfo 付与
            last = last.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)   # aware
        if (now - last) < timedelta(days=60):
            # ── キャッシュをそのまま返却 ─────────────────────
            return jsonify({
                'ok': True,
                'lemmas':   int(row.vocab_lemmas),
                'cefr':     row.vocab_cefr or '',
                'comment':  row.vocab_comment or '',
                'updated':  last.isoformat()
            })

    # 3) 60 日を超えている（または updated が None）→ 再計算を促す
    return jsonify({'ok': False})


def norm(w:str)->str:
    """空白除去＋小文字化"""
    return re.sub(r'\s+', '', w or '').lower()


@app.route('/api/personal_vocab/candidates', methods=['POST'])
def vocab_candidates():
    """
    1. users.vocab_* を参照して学習者レベルを取得
    2. パーソナル辞書の全単語を known_set に格納
    3. そこからランダム 50 語だけ GPT へヒントとして渡す
    4. GPT (o4-mini) に EXACTLY 5 語を要求
    5. サーバ側で重複除去 ─ 足りなければ最大 3 回まで再試行
    """
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403
    uid = session['user_id']
    flavor = (request.json or {}).get('flavor', 'general').lower()

    # ── ① プロフィール取得 ───────────────────────────
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT vocab_lemmas, vocab_cefr
          FROM dbo.users WHERE id = ?
    """, uid)
    row = cur.fetchone()
    if not row or row.vocab_lemmas is None:
        cur.close(); conn.close()
        return jsonify({'error': 'profile_required'}), 409
    lemmas_est = int(row.vocab_lemmas)
    cefr_est   = row.vocab_cefr or 'B1'

    cur.execute("""
                WITH latest AS (SELECT vocab_id,
                                       ISNULL(ef, 0) AS ef,
                                       ROW_NUMBER()     OVER (PARTITION BY vocab_id ORDER BY review_time DESC) rn
                                FROM dbo.vocab_reviews
                                WHERE user_id = ?)
                SELECT TOP 50 vi.word, lat.ef
                FROM dbo.courses c
                         JOIN dbo.vocab_items vi ON vi.course_id = c.id
                         LEFT JOIN latest lat ON lat.vocab_id = vi.id AND lat.rn = 1
                WHERE c.owner_user_id = ?
                  AND c.category_id = 4
                  AND vi.word IS NOT NULL
                ORDER BY ISNULL(lat.ef, 0) ASC, NEWID() -- EF が低い順、同点はランダム
                """, uid, uid)
    rows = cur.fetchall()
    known_set = {norm(r.word) for r in rows}  # 低 EF 50 語を既知セットにも使う
    cur.close();
    conn.close()

    # rows は EF が低い順に最大 50 行だけ返ってくる
    sample_words = [r.word for r in rows]  # そのまま配列化
    sample_hint = ', '.join(sample_words)  # GPT へのヒント

    # ── ③ GPT プロンプト（flavor 追加） ────────────────
    flavor_note = {
        'idiom':'Focus on idioms or phrasal verbs. ',
        'slang':'Focus on informal Gen-Z slang. ',
        'business':'Focus on words used in a business context and make sure they match the CEFR target and lemma-size guidance described above. '
    }.get(flavor, '')

    base_prompt = (
        "You are an English-vocabulary tutor.\n"
        f"The learner knows about {lemmas_est:,} lemmas and is around CEFR {cefr_est}.\n"
        + flavor_note +
        "Avoid duplicates from the list below and propose EXACTLY 5 new words "
        "at the same level or slightly higher.\n"
        "Return ONE JSON object ONLY: {\"words\": [\"...\", ...]}.\n\n"
        f"Known words sample: {sample_hint}"
    )

    # ── ④ 最大 3 回まで試行 ──────────────────────────
    for attempt in range(3):
        # ① セッションからユーザー ID
        uid = session.get('user_id', 0)

        # ② safe_chat で呼び出し
        try:
            rsp = safe_chat(
                user_id=uid,
                client=o4_client,
                deployment=O4_DEPLOY,
                messages=[{"role": "system", "content": base_prompt}],
                max_completion_tokens=2000,
                response_format={"type": "json_object"}
            )
        except RuntimeError as e:
            # 上限超過 → その場で 429 を返す
            return jsonify({"error": str(e)}), 429
        try:
            words_raw = json.loads(rsp.choices[0].message.content).get('words', [])
        except Exception:
            # パース失敗 → もう一度
            time.sleep(0.4); continue

        # ── ⑤ 重複フィルタ ─────────────────────────
        clean, seen = [], set()
        for w in words_raw:
            nw = norm(w)
            if nw and nw not in known_set and nw not in seen:
                clean.append(w.strip()); seen.add(nw)
            if len(clean) == 5:
                break

        if len(clean) == 5:      # 成功
            return jsonify({'ok': True, 'words': clean})

        # 失敗 → 少し待って再リクエスト
        time.sleep(0.4)

    # 3 回失敗
    return jsonify({'error': 'duplicate'}), 409


# -----------------------------------------------------------------
# /api/personal_vocab/examples  POST  {"words":["..."]}
#   → 例文リストを返しつつ personal 辞書に自動追加
# -----------------------------------------------------------------
@app.route('/api/personal_vocab/examples', methods=['POST'])
def vocab_examples():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403
    uid = session['user_id']
    words = (request.get_json(force=True) or {}).get('words', [])

    # ------- バリデーション --------------------------------------
    if (not isinstance(words, list) or
        not all(isinstance(w, str) for w in words) or
        len(words) == 0 or len(words) > 10):
        return jsonify({'error': 'bad_request'}), 400

    # ------- GPT で例文生成 --------------------------------------
    prompt = (
        "Give ONE concise (≤15 words) example sentence for each word below.\n"
        "Return ONE JSON object mapping words to sentences.\n\n"
        f"Words: {', '.join(words)}"
    )

    uid = session.get('user_id', 0)  # 未ログインなら 0

    try:
        rsp = safe_chat(
            user_id=uid,
            client=o4_client,
            deployment=O4_DEPLOY,
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=3000,
            response_format={"type": "json_object"}
        )
    except RuntimeError as e:
        # トークン上限などで safe_chat が例外を出した場合
        return jsonify({"error": str(e)}), 429
    try:
        obj = json.loads(rsp.choices[0].message.content)
    except Exception:
        return jsonify({'error': 'parse'}), 502

    examples = [
        {"word": w, "sentence": (obj.get(w) or '').strip()}
        for w in words if (obj.get(w) or '').strip()
    ]
    if not examples:
        return jsonify({'error': 'no_examples'}), 502

    # ------- personal 辞書に自動追加 -----------------------------
    course_id, course_name = ensure_personal_course(uid)  # helper 既存
    conn = get_db_connection(); cur = conn.cursor()

    for ex in examples:
        # ① すでに登録済みならスキップ
        cur.execute("""
            SELECT 1 FROM dbo.vocab_items
             WHERE course_id = ? AND word = ?
        """, course_id, ex['word'])
        if cur.fetchone():
            continue

        # ② INSERT
        cur.execute("""
            INSERT INTO dbo.vocab_items (course_id, course, sentence, word)
            VALUES (?, ?, ?, ?)
        """, course_id, course_name, ex['sentence'], ex['word'])

    conn.commit(); cur.close(); conn.close()

    return jsonify({'ok': True, 'examples': examples})


@app.route('/boost')
def boost_page():
    """Boost – AI が単語5件を提案するページ"""
    if 'user_id' not in session:
        return redirect(url_for('login_user'))   # 未ログインならログインへ
    return render_template('boost.html',
                           user_name=session.get('user_name', ''))


@app.route('/settings')
def settings_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('settings.html')

# ----------------------------------------------------------
# ② /api/user_summary  … JSON でプロフィール＋学習数
# ----------------------------------------------------------
@app.route('/api/user_summary')
def user_summary():
    if 'user_id' not in session:
        return jsonify({'error':'login'}), 403
    uid = session['user_id']

    conn = get_db_connection(); cur = conn.cursor()
    # ユーザー基本＋語彙プロフィール
    cur.execute("""
        SELECT user_id, name, email,
               vocab_lemmas, vocab_cefr, vocab_comment
          FROM dbo.users WHERE id = ?
    """, uid)
    row = cur.fetchone()

    # vocab_reviews でユニーク語数
    cur.execute("""
        SELECT COUNT(DISTINCT vocab_id)
          FROM dbo.vocab_reviews
         WHERE user_id = ?
    """, uid)
    studied = cur.fetchone()[0] or 0

    # ── C) パーソナル辞書 (category_id=4) の総語数 ────────────
    cur.execute("""
        SELECT COUNT(*)
          FROM dbo.vocab_items vi
          JOIN dbo.courses c ON c.id = vi.course_id
         WHERE c.owner_user_id = ?
           AND c.category_id   = 4
    """, uid)
    personal_cnt = cur.fetchone()[0] or 0

    cur.close(); conn.close()

    return jsonify({
        'ok': True,
        'user_id':   row.user_id,
        'name':      row.name,
        'email':     row.email,
        'lemmas':    row.vocab_lemmas,
        'cefr':      row.vocab_cefr,
        'comment':   row.vocab_comment,
        'studied':   studied,
        'personal_dict': personal_cnt  # ★ 追加フィールド
    })

# ─────────────────────────────────────────────────────
#  /api/change_email  : パスワード確認つきメール変更
# ─────────────────────────────────────────────────────
@app.route('/api/change_email', methods=['POST'])
def change_email():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403

    uid          = session['user_id']
    current_pw   = request.form.get('password', '')
    new_email    = (request.form.get('new_email', '')).strip().lower()

    # --- 0) 形式チェック -------------------------------------------------
    import re, secrets
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', new_email):
        return jsonify({'error': 'bad_email'}), 400

    conn = get_db_connection(); cur = conn.cursor()

    # --- 1) 現在パスワード検証 ------------------------------------------
    cur.execute("SELECT password_hash, email FROM dbo.users WHERE id = ?", uid)
    row = cur.fetchone()
    if not row:
        return jsonify({'error': 'not_found'}), 404
    if not check_password(current_pw, row.password_hash):
        return jsonify({'error': 'wrong_pw'}), 401

    if new_email == row.email:
        return jsonify({'error': 'same'}), 409      # 変更なし

    # --- 2) 既に使われていないか確認 -----------------------------------
    cur.execute("SELECT 1 FROM dbo.users WHERE email = ?", new_email)
    if cur.fetchone():
        return jsonify({'error': 'duplicate'}), 409

    # --- 3) email 更新・再認証フラグ・token 再発行 ----------------------
    token = secrets.token_urlsafe(32)
    cur.execute("""
        UPDATE dbo.users
           SET email = ?, is_email_verified = 0, verify_token = ?
         WHERE id = ?
    """, new_email, token, uid)
    conn.commit()

    # --- 4) 確認メール送信 ---------------------------------------------
    verify_link = f"{VERIFY_BASE_URL}?token={token}"
    mail.send(Message(
        subject='[Polyagent AI] Confirm your new e-mail address',
        recipients=[new_email],
        body=f"""Hi {session.get('user_name','')},

You (or someone) requested to change the e-mail associated with your Polyagent AI account.

Please verify your new address by clicking the link below:

{verify_link}

If you did not request this, you can safely ignore this message.
"""
    ))

    cur.close(); conn.close()
    return jsonify({'ok': True})


# settings 用サブページ
@app.route('/settings/change_email')
def change_email_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('change_email.html')


# --------------------------------------------------
#  /dictionary  画面本体
# --------------------------------------------------
@app.route('/dictionary')
def dictionary_page():
    if 'user_id' not in session:
        return redirect(url_for('login_user'))
    return render_template('dictionary.html')

# --------------------------------------------------
#  /api/personal_vocab/list?sort=ef|created  ← JSON
# --------------------------------------------------
@app.route('/api/personal_vocab/list')
def personal_vocab_list():
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403
    uid   = session['user_id']
    sort  = request.args.get('sort', 'created')   # 既定 = 追加日 DESC

    # ── ① 並べ替えオプションに next を追加 ──
    order = {
        'ef'      : 'ISNULL(lat.ef,2.5) DESC',
        'created' : 'vi.created_at DESC',
        'word'    : 'vi.word ASC',
        # next_review が NULL の行は “遠い未来” として末尾に回す
        'next'    : 'ISNULL(lat.next_review, \'9999-12-31\') ASC'
    }.get(sort, 'vi.created_at DESC')

    conn = get_db_connection(); cur = conn.cursor()
    cur.execute(f"""
        /* 最新レビュー (EF / next_review) を JOIN して一発で返す */
        WITH latest AS (
            SELECT vocab_id,
                   ef,
                   next_review,
                   ROW_NUMBER() OVER (PARTITION BY vocab_id
                                      ORDER BY review_time DESC) rn
              FROM dbo.vocab_reviews
             WHERE user_id = ?
        )
        SELECT vi.id,
               vi.word,
               vi.sentence,
               vi.created_at,
               ISNULL(lat.ef, 2.5)            AS ef,
               lat.next_review                AS next_review
          FROM dbo.courses c
          JOIN dbo.vocab_items vi ON vi.course_id = c.id
          LEFT JOIN latest      lat ON lat.vocab_id = vi.id AND lat.rn = 1
         WHERE c.owner_user_id = ? AND c.category_id = 4
         ORDER BY {order}
    """, uid, uid)

    rows = [dict(
                id       = r.id,
                word     = r.word,
                sentence = r.sentence,
                created  = (r.created_at.isoformat() if r.created_at else ''),
                ef       = float(r.ef),
                next_review = (r.next_review.isoformat() if r.next_review else '')
            )
            for r in cur]
    cur.close(); conn.close()
    return jsonify({'ok': True, 'items': rows})


@app.route('/api/personal_vocab/master/<int:vocab_id>', methods=['POST'])
def pvocab_master(vocab_id: int):
    """
    『覚えた！』ボタン：
      • self_score = 5
      • ef         = 30.0
      • next_review = 90 日後
    を vocab_reviews に 1 行 INSERT
    """
    if 'user_id' not in session:
        return jsonify({'error': 'login'}), 403
    uid = session['user_id']

    conn = get_db_connection(); cur = conn.cursor()
    try:
        # ← 90 日後 (UTC) を計算
        cur.execute("SELECT DATEADD(day, 90, GETUTCDATE())")
        next90 = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO dbo.vocab_reviews
                   (vocab_id, review_time, self_score,
                    test_score, ef, next_review, user_id)
            VALUES (?, GETUTCDATE(),         -- 今回のレビュー時刻
                    5,                      -- self_score = MAX
                    NULL,                   -- test_score 使わない
                    30.0,                   -- ★ EF = 30
                    ?,                      -- ★ 90 日後
                    ?);                     -- user_id
        """, vocab_id, next90, uid)

        conn.commit()
        return jsonify({'ok': True})
    except Exception as e:
        conn.rollback()
        print('[MASTER ERR]', e)
        return jsonify({'error': 'db'}), 500
    finally:
        cur.close(); conn.close()


@app.route('/api/personal_vocab/delete/<int:vocab_id>', methods=['POST'])
def pvocab_delete(vocab_id):
    if 'user_id' not in session:
        return jsonify({'error':'login'}), 403
    uid = session['user_id']

    conn = get_db_connection(); cur = conn.cursor()
    # パーソナル辞書 (category_id = 4) か & 自分の所有か を確認して削除
    cur.execute("""
        DELETE vi
          FROM dbo.vocab_items vi
          JOIN dbo.courses c ON c.id = vi.course_id
         WHERE vi.id = ? AND c.owner_user_id = ? AND c.category_id = 4
    """, vocab_id, uid)
    conn.commit(); cur.close(); conn.close()
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=True)

