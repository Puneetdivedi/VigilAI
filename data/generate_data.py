"""
Generates synthetic sensor data simulating industrial machines.
Inserts data into SQLite via SQLAlchemy and saves to CSV.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db.models import SensorReading, SessionLocal, engine

def generate_synthetic_data(num_rows: int = 10000, fault_ratio: float = 0.15) -> pd.DataFrame:
    """
    Generates synthetic sensor data with given rows and fault ratio.
    """
    np.random.seed(42)
    machines = [f"MACH_{i:03d}" for i in range(1, 6)]
    
    data = []
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30)
    
    timestamps = [start_time + timedelta(seconds=(end_time - start_time).total_seconds() * i / num_rows) for i in range(num_rows)]
    
    num_faults = int(num_rows * fault_ratio)
    fault_indices = np.random.choice(num_rows, num_faults, replace=False)
    
    for i in range(num_rows):
        is_fault = i in fault_indices
        machine = np.random.choice(machines)
        
        if is_fault:
            # Fault condition: high vibration + abnormal temp or pressure
            fault_type = np.random.choice(["bearing_fault", "overheating", "pressure_anomaly"])
            if fault_type == "bearing_fault":
                vibration = np.random.normal(loc=25.0, scale=5.0)
                temp = np.random.normal(loc=70.0, scale=10.0)
                rpm = np.random.normal(loc=2900, scale=100)
                pressure = np.random.normal(loc=1.2, scale=0.1)
            elif fault_type == "overheating":
                vibration = np.random.normal(loc=10.0, scale=2.0)
                temp = np.random.normal(loc=110.0, scale=15.0)
                rpm = np.random.normal(loc=3000, scale=50)
                pressure = np.random.normal(loc=1.2, scale=0.1)
            else: # pressure_anomaly
                vibration = np.random.normal(loc=12.0, scale=3.0)
                temp = np.random.normal(loc=65.0, scale=5.0)
                rpm = np.random.normal(loc=3000, scale=50)
                pressure = np.random.normal(loc=2.5, scale=0.4)
            label = fault_type
        else:
            # Normal operating conditions
            vibration = np.random.normal(loc=5.0, scale=1.5)
            temp = np.random.normal(loc=60.0, scale=5.0)
            rpm = np.random.normal(loc=3000, scale=50)
            pressure = np.random.normal(loc=1.2, scale=0.1)
            label = "normal"
            
        data.append({
            "timestamp": timestamps[i],
            "machine_id": machine,
            "vibration": max(0, vibration),
            "temperature": max(0, temp),
            "rpm": max(0, rpm),
            "pressure": max(0, pressure),
            "label": label # For training purposes, saved to CSV only, not DB sensor_readings
        })
        
    df = pd.DataFrame(data)
    return df

def save_to_db(df: pd.DataFrame):
    """
    Saves dataframe to SQLite database using SQLAlchemy.
    """
    session = SessionLocal()
    try:
        # Clear existing data for freshness during generation
        session.query(SensorReading).delete()
        
        readings = [
            SensorReading(
                machine_id=row["machine_id"],
                vibration=row["vibration"],
                temperature=row["temperature"],
                rpm=row["rpm"],
                pressure=row["pressure"],
                timestamp=row["timestamp"].to_pydatetime()
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(readings)
        session.commit()
        print(f"Inserted {len(readings)} rows into database.")
    except Exception as e:
        session.rollback()
        print(f"Error saving to DB: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    print("Generating synthetic data...")
    os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)
    df = generate_synthetic_data(10000, 0.15)
    
    csv_path = os.path.join(os.path.dirname(__file__), "sensor_readings.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    
    save_to_db(df)
    print("Done.")
