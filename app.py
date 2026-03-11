"""
Simple Flask-based GUI to start/stop the scripts in this repository and
view their output logs.  The application runs locally and is intended to
be accessed in a browser at http://localhost:5000/ (or via the dev
container forwarded port).

Usage:
    python app.py

Then open your browser and navigate to http://127.0.0.1:5000/.

The GUI shows a table of available scripts, a start/stop button for each,
and links to view the corresponding log file. Processes are launched
with subprocess.Popen and kept in a global dictionary. Logs are written
to `logs/<script>.log`.

This is a minimal management interface; for more advanced control you
could replace the process logic with Celery, RQ, or another task queue.
"""

import os
import subprocess
from flask import Flask, render_template_string, redirect, url_for, send_from_directory, abort

app = Flask(__name__)

# list of scripts we know about (relative to repo root)
SCRIPTS = [
    'backtest.py',
    'backtest_002.py',
    'bot.py',
    'bot_1.py',
    'ict.py',
    'ict_fibo_volume_strategy.py',
    'newtest.py',
    'tetet.py',
    'highest_profit_factor.py',
]

processes = {}  # script -> subprocess.Popen instance
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

INDEX_TEMPLATE = r"""
<!doctype html>
<html>
<head>
    <title>KI Trading Bot Control</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        button { padding: 4px 8px; }
    </style>
</head>
<body>
    <h1>KI Trading Bot Control Panel</h1>
    <table>
        <tr><th>Script</th><th>Status</th><th>Actions</th><th>Log</th></tr>
        {% for script, running in status %}
        <tr>
            <td>{{ script }}</td>
            <td>{{ 'running' if running else 'stopped' }}</td>
            <td>
                {% if running %}
                    <a href="{{ url_for('stop_script', script=script) }}"><button>Stop</button></a>
                {% else %}
                    <a href="{{ url_for('start_script', script=script) }}"><button>Start</button></a>
                {% endif %}
            </td>
            <td>
                <a href="{{ url_for('view_log', script=script) }}">view</a>
            </td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route('/')
def index():
    status = []
    for script in SCRIPTS:
        proc = processes.get(script)
        running = proc is not None and proc.poll() is None
        status.append((script, running))
    return render_template_string(INDEX_TEMPLATE, status=status)


@app.route('/start/<path:script>')
def start_script(script):
    if script not in SCRIPTS:
        abort(404)
    proc = processes.get(script)
    if proc and proc.poll() is None:
        # already running
        return redirect(url_for('index'))

    logpath = os.path.join(LOG_DIR, os.path.basename(script) + '.log')
    logfile = open(logpath, 'a')
    # start the script in a new process
    processes[script] = subprocess.Popen(
        ['python', script],
        cwd=os.getcwd(),
        stdout=logfile,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return redirect(url_for('index'))


@app.route('/stop/<path:script>')
def stop_script(script):
    proc = processes.get(script)
    if proc and proc.poll() is None:
        proc.terminate()
    return redirect(url_for('index'))


@app.route('/log/<path:script>')
def view_log(script):
    if script not in SCRIPTS:
        abort(404)
    filename = os.path.basename(script) + '.log'
    return send_from_directory(LOG_DIR, filename, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
