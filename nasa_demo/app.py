import os
import json
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template
import pandas as pd

# --- Add the python sub-directory to the path ---
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(_PROJECT_ROOT / "atomic-lang-model" / "python"))

# --- Import the ALM components ---
from atomic_lang_model_python import validate_mission_log
from tiny_lm import ProbGrammar

app = Flask(__name__)

# --- Path to the new, large mission log ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_BASE_DIR, "data", "mission_log_large.csv")

# --- Grammar Explanation for the UI ---
# This text will be displayed on the page to explain the demo to JPL.
GRAMMAR_EXPLANATION = {
    "title": "About the Telemetry Language",
    "intro": "The Atomic Language Model (ALM) does not just find statistical outliers; it parses the stream of events against a formal grammar of operations. An 'anomaly' is an 'ungrammatical' sequence of events that violates the rules of the mission.",
    "rules": [
        {
            "name": "Rule 1: Context-Sensitive Commands",
            "desc": "A command must be valid within its current context (DRIVE, SCIENCE, or STANDBY). For example, a `MOTOR_CMD_START` is grammatical in the `CTX_DRIVE` context but not `CTX_SCIENCE`."
        },
        {
            "name": "Rule 2: State Dependencies",
            "desc": "Certain states are only valid if preceded by a specific command. For example, a `VOLTAGE_SPIKE` is only considered 'grammatical' if it occurs immediately after a command like `MOTOR_CMD_START`."
        },
        {
            "name": "Rule 3: Invalid Sequences",
            "desc": "Any other sequence of two events that does not have a valid `Merge` operation in the grammar is considered ungrammatical. For example, a `VOLTAGE_SPIKE` during `CTX_STANDBY` is an anomaly because there is no rule to parse it."
        }
    ],
    "conclusion": "This demo showcases how the ALM can detect complex, context-dependent anomalies that simple thresholding would miss, providing a higher level of assurance for mission-critical systems."
}


def load_data():
    """Load mission log data from the local CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', grammar=GRAMMAR_EXPLANATION)

@app.route('/get_mission_log')
def get_mission_log():
    """
    This endpoint loads the mission log, validates it with the ALM Rust Core,
    and returns the data and identified anomalies.
    """
    df = load_data()
    if df is None:
        return jsonify({"error": f"Data file not found at {DATA_FILE}"}), 404

    # --- Prepare data for the ALM Core ---
    # We send the raw event names to the ALM. The grammar rules in Rust
    # will determine if the sequence is valid.
    log_events = df['event'].tolist()
    
    # --- Call the Rust Core for Formal Validation ---
    detected_anomalies = validate_mission_log(log_events)

    # --- Call the Python Core for Probabilistic Analysis ---
    prob_model = ProbGrammar()
    log_sentence = " ".join(log_events)
    # We calculate a "surprise" score. A lower score is better.
    # A true perplexity is more complex; we use the model's average log probability as a proxy.
    surprise_score = prob_model.calculate_sentence_probability(log_sentence)
    
    # --- Prepare data for visualization ---
    log_data = df.to_dict(orient='records')

    return jsonify({
        "log_data": log_data,
        "anomalies": detected_anomalies,
        "surprise_score": f"{surprise_score:.4f}",
        "summary": f"Parsed {len(log_events)} events. Found {len(detected_anomalies)} ungrammatical sequences."
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)