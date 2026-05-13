import csv
import random
import os

# --- Configuration ---
# Get the directory of the current script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(_BASE_DIR, 'data', 'mission_log_large.csv')
NUM_EVENTS = 500
ANOMALY_PROBABILITY = 0.05

# --- Event Definitions ---
# (subsystem, event, normal_value_range, next_context)
NORMAL_OPS = {
    'DRIVE': [
        ('POWER', 'MOTOR_CMD_START', (1, 1), 'DRIVE'),
        ('POWER', 'CURRENT_DRAW', (2.5, 4.0), 'DRIVE'),
        ('MOBILITY', 'WHEEL_RPM', (50, 70), 'DRIVE'),
        ('THERMAL', 'TEMP_MOTOR', (40, 60), 'DRIVE'),
        ('POWER', 'MOTOR_CMD_STOP', (1, 1), 'STANDBY'),
    ],
    'SCIENCE': [
        ('POWER', 'INSTRUMENT_PWR_ON', (1, 1), 'SCIENCE'),
        ('SCIENCE', 'SPECTROMETER_READ', (100, 200), 'SCIENCE'),
        ('POWER', 'CURRENT_DRAW', (1.0, 1.5), 'SCIENCE'),
        ('THERMAL', 'TEMP_INSTRUMENT', (20, 30), 'SCIENCE'),
        ('POWER', 'INSTRUMENT_PWR_OFF', (1, 1), 'STANDBY'),
    ],
    'STANDBY': [
        ('POWER', 'CURRENT_DRAW', (0.4, 0.6), 'STANDBY'),
        ('THERMAL', 'TEMP_MOTOR', (20, 30), 'STANDBY'),
        ('THERMAL', 'TEMP_INSTRUMENT', (15, 25), 'STANDBY'),
        # Add transitions to other states
        ('POWER', 'MOTOR_CMD_START', (1, 1), 'DRIVE'),
        ('POWER', 'INSTRUMENT_PWR_ON', (1, 1), 'SCIENCE'),
    ]
}

# --- Anomaly Definitions ---
# (subsystem, event, value, context, description)
ANOMALIES = [
    ('POWER', 'VOLTAGE_SPIKE', 12.1, 'STANDBY', "Voltage spike during standby is ungrammatical."),
    ('POWER', 'CURRENT_DRAW', 3.5, 'STANDBY', "High current draw during standby is ungrammatical."),
    ('MOBILITY', 'WHEEL_RPM', 60, 'SCIENCE', "Wheel movement during science observation is ungrammatical."),
    ('THERMAL', 'TEMP_MOTOR', 75, 'STANDBY', "Motor overheating during standby is ungrammatical."),
    ('POWER', 'INSTRUMENT_PWR_ON', 1, 'DRIVE', "Instrument power-on during drive is ungrammatical."),
]

def generate_event(timestamp, current_context):
    """Generates a single event, possibly an anomaly."""
    if random.random() < ANOMALY_PROBABILITY:
        # Inject an anomaly
        subsystem, event, value, context, desc = random.choice(ANOMALIES)
        # We use the anomaly's context to make it more realistic
        return {'timestamp': timestamp, 'subsystem': subsystem, 'event': event, 'value': value, 'context': context}, context

    # Generate a normal event
    # Ensure there's a way to transition out of the current state
    possible_events = NORMAL_OPS.get(current_context, NORMAL_OPS['STANDBY'])
    subsystem, event, value_range, next_context = random.choice(possible_events)
    value = round(random.uniform(value_range[0], value_range[1]), 2)
    
    return {'timestamp': timestamp, 'subsystem': subsystem, 'event': event, 'value': value, 'context': current_context}, next_context

def main():
    """Generates the mission log CSV."""
    header = ['timestamp', 'subsystem', 'event', 'value', 'context']
    log_entries = []
    current_context = 'STANDBY'
    
    print(f"Generating {NUM_EVENTS} events for {OUTPUT_FILE}...")

    for i in range(NUM_EVENTS):
        timestamp = 1000 + i
        event_data, new_context = generate_event(timestamp, current_context)
        log_entries.append(event_data)
        current_context = new_context

    try:
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(log_entries)
        print(f"Successfully generated {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    main()
