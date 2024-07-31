# login.py

from flask import Flask, session, redirect, url_for, render_template, request, flash
import csv
import bcrypt
import os

# Define the path to the resources folder
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), 'resources')

def load_users():
    """Load users from a CSV file."""
    users = []
    with open(os.path.join(RESOURCE_PATH, 'users.csv'), newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            users.append(row)
    return users

def login_user(app):
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password'].encode('utf-8')

            users = load_users()
            for user in users:
                if user['email'] == email:
                    if bcrypt.checkpw(password, user['password'].encode('utf-8')):
                        session['user'] = user['name']
                        flash('Login successful!', 'success')
                        return redirect(url_for('dashboard'))
                    else:
                        flash('Invalid credentials', 'danger')
                        return render_template('login.html', message='Invalid credentials')
            flash('User not found', 'danger')
            return render_template('login.html', message='User not found')

        return render_template('login.html', message="")

    @app.route('/logout')
    def logout():
        session.pop('user', None)
        flash('You have been logged out', 'info')
        return redirect(url_for('login'))

def init_app(app):
    login_user(app)
