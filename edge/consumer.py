import json
from datetime import datetime
import paho.mqtt.client as mqtt

# --- Configuration ---
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "telemetry/car1"
REQUIRED_FIELDS = ["vehicle_id", "speed", "rpm", "engine_temp"]
LOG_FILE = "logs/telemetry.jsonl"

def process_telemetry(data):
    """Validates and processes the incoming telemetry dictionary."""
    
    # 1. Validation: Check if all required fields are present
    for field in REQUIRED_FIELDS:
        if field not in data:
            print(f"⚠️ Validation Error: Missing field '{field}' in {data}")
            return None
    
    # 2. Timestamping: Add the current time
    # isoformat() makes it look like "2026-08-16T12:00:00.123456"
    data["timestamp"] = datetime.now().isoformat()
    
    return data

def log_telemetry(data):
    """Appends the processed telemetry to the JSONL log file."""
    # Open in 'a' (append) mode so we don't overwrite existing data
    with open(LOG_FILE, 'a') as f:
        # Convert the dictionary back to a JSON string and add a newline
        f.write(json.dumps(data) + '\n')

# --- MQTT Callbacks ---

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback triggered when the client connects to the broker."""
    if reason_code == 0:
        print(f"✅ Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        # Subscribe to the topic once connected
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to topic: {TOPIC}")
    else:
        print(f"❌ Failed to connect, reason code: {reason_code}")

def on_message(client, userdata, message):
    """Callback triggered whenever a message is received on a subscribed topic."""
    
    # The message payload is raw bytes, so we decode it to a string
    raw_payload = message.payload.decode('utf-8')
    
    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(raw_payload)
        
        # Process, validate, and timestamp the data
        processed_data = process_telemetry(data)
        
        if processed_data:
            print(f"📥 Processed: {processed_data}")
            
            # 3. Log the processed data to our file
            log_telemetry(processed_data)
            
    except json.JSONDecodeError:
        print(f"⚠️ Error: Received malformed JSON: {raw_payload}")

def main():
    print("🖥️ Starting Edge Consumer...")
    
    # Create the MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    
    # Assign our callback functions
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        # Connect to the Mosquitto broker
        client.connect(BROKER_HOST, BROKER_PORT)
        
        # Start the background loop that listens for messages
        # This will block forever until Ctrl+C is pressed
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Consumer stopped by user.")
    finally:
        client.disconnect()
        print("Disconnected from broker.")

if __name__ == "__main__":
    main()
