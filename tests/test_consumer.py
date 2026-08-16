import sys
import os

# Add the project root to the Python path so we can import from the 'edge' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.consumer import process_telemetry

def test_valid_telemetry():
    """Test that valid telemetry is processed successfully and timestamped."""
    valid_data = {
        "vehicle_id": "car_01",
        "speed": 100,
        "rpm": 3000,
        "engine_temp": 90
    }
    
    result = process_telemetry(valid_data)
    
    # It should not return None
    assert result is not None, "Valid data should not be rejected."
    
    # It should have added a timestamp
    assert "timestamp" in result, "Processed data should contain a timestamp."
    
    print("✅ test_valid_telemetry passed!")

def test_missing_field():
    """Test that telemetry missing a required field is rejected (returns None)."""
    invalid_data = {
        "vehicle_id": "car_01",
        # "speed" is missing!
        "rpm": 3000,
        "engine_temp": 90
    }
    
    result = process_telemetry(invalid_data)
    
    # It should return None because validation failed
    assert result is None, "Data missing a required field should be rejected."
    
    print("✅ test_missing_field passed!")

if __name__ == "__main__":
    print("Running consumer tests...")
    test_valid_telemetry()
    test_missing_field()
    print("🎉 All tests passed!")
