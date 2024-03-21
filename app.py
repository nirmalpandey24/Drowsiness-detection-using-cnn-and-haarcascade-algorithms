from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_script')
def start_script():
    try:
        subprocess.Popen(['python', 'drowsiness_detection.py'])  # Replace 'your_script.py' with the actual filename
        return "Script started successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
