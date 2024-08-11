from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import check_password_hash
import csv
import os

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY')

# Path to the CSV file with user credentials
USERS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'users.csv')

# Load user data from CSV
def load_user_data():
    users = {}
    with open(USERS_CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_id = row['id']
            users[user_id] = {
                'name': row['name'],
                'password': row['password']
            }
    return users

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']

        # Load user data from CSV
        users = load_user_data()

        # Check if user_id exists and password matches
        if user_id in users and check_password_hash(users[user_id]['password'], password):
            session['user_id'] = user_id
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='IDまたはパスワードが違います。')

    return render_template('login.html', message='IDとパスワードを入力してください')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user_id=session['user_id'])

if __name__ == '__main__':
    app.run(debug=True)
