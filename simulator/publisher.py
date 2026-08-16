import json
import time
import random
import paho.mqtt.client as mqtt

# --- Configuration ---
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "telemetry/car1"
VEHICLE_ID = "car_01"

def generate_telemetry():
    """Generates fake, randomized telemetry data for a vehicle."""
    return {
        "vehicle_id": VEHICLE_ID,
        "speed": random.randint(40, 140),          # Speed in km/h
        "rpm": random.randint(2000, 6000),         # Engine RPM
        "engine_temp": random.randint(80, 110)     # Temperature in Celsius
    }

def main():
    print(f"🚗 Starting Vehicle Simulator ({VEHICLE_ID})...")
    
    # 1. Create the MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    
    try:
        # 2. Connect to the Mosquitto broker
        client.connect(BROKER_HOST, BROKER_PORT)
        print(f"✅ Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        
        # 3. Start the infinite loop to send data continuously
        while True:
            # Generate the data dictionary
            data = generate_telemetry()
            
            # 4. Convert the Python dictionary to a JSON string
            payload = json.dumps(data)
            
            # 5. Publish to the topic
            client.publish(TOPIC, payload)
            
            print(f"Published to {TOPIC}: {payload}")
            
            # 6. Wait 1 second before sending the next reading
            time.sleep(1)
            
    except KeyboardInterrupt:
        # 7. Handle Ctrl+C gracefully
        print("\n🛑 Simulation stopped by user.")
    finally:
        # Cleanly disconnect from the broker
        client.disconnect()
        print("Disconnected from broker.")

if __name__ == "__main__":
    main()
