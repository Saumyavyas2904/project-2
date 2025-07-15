import websocket
import json
import os
from psycopg2 import connect, Error
import numpy as np
from datetime import datetime
import threading
import time

# Function to establish database connection with retry
def get_db_connection():
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable not set")
            conn = connect(db_url)
            cursor = conn.cursor()
            print("Connected to PostgreSQL successfully!")
            return conn, cursor
        except Error as e:
            print(f"Error connecting to PostgreSQL (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(2 ** retry_count)  # Exponential backoff
    raise Exception("Failed to connect to PostgreSQL after multiple attempts")

# Initial connection
try:
    conn, cursor = get_db_connection()
    
    # Get the last sample number
    cursor.execute("SELECT COALESCE(MAX(sample), 0) FROM samples")
    last_sample = cursor.fetchone()[0]
    sample_count = last_sample + 1
    print(f"Starting from sample {sample_count}")
except Exception as e:
    print(f"Critical error: {e}")
    exit()

# Get sensor ID
sensor_id = int(os.getenv("SENSOR_ID", "1"))
if sensor_id not in [1, 2, 3, 4, 5]:
    print("Invalid sensor ID from environment. Using sensor ID 1.")
    sensor_id = 1

# Calculation settings
VEL_WINDOW = 10
DT = 0.015
HPF_ALPHA = 0.84
LPF_BETA = 0.32
RESET_INTERVAL = 20
ACCEL_LIMIT = 49050

# Buffers and state
vel_buffer_x = [0] * VEL_WINDOW
vel_buffer_y = [0] * VEL_WINDOW
vel_buffer_z = [0] * VEL_WINDOW
disp_buffer_x = [0] * VEL_WINDOW
disp_buffer_y = [0] * VEL_WINDOW
disp_buffer_z = [0] * VEL_WINDOW
vel_index = 0
vel_x, vel_y, vel_z = 0, 0, 0
disp_x, disp_y, disp_z = 0, 0, 0
prev_acc_x, prev_acc_y, prev_acc_z = 0, 0, 0
prev_vel_x, prev_vel_y, prev_vel_z = 0, 0, 0
prev_hpf_x, prev_hpf_y, prev_hpf_z = 0, 0, 0
prev_lpf_x, prev_lpf_y, prev_lpf_z = 0, 0, 0
prev_input_x, prev_input_y, prev_input_z = 0, 0, 0
sample_counter = 0
all_data = []

# WebSocket functions
def on_message(ws, message):
    global vel_index, vel_x, vel_y, vel_z, disp_x, disp_y, disp_z
    global prev_acc_x, prev_acc_y, prev_acc_z, prev_vel_x, prev_vel_y, prev_vel_z
    global prev_hpf_x, prev_hpf_y, prev_hpf_z, prev_lpf_x, prev_lpf_y, prev_lpf_z
    global prev_input_x, prev_input_y, prev_input_z, sample_counter, sample_count
    print(f"Received Raw: {message}")
    try:
        if message in ["Authenticated", "Authentication failed", "Resetting sensor..."]:
            print(f"Skipping message: {message}")
            return

        data = json.loads(message)
        print(f"Parsed Data: {data}")

        if 'a' in data and 'vib' in data and isinstance(data['a'], list) and len(data['a']) == 3 and isinstance(data['vib'], list) and len(data['vib']) == 3:
            sample_count += 1
            sample_counter += 1
            x = float(data['a'][0])
            y = float(data['a'][1])
            z = float(data['a'][2])
            h_vib = float(data['vib'][0])
            v_vib = float(data['vib'][1])
            a_vib = float(data['vib'][2])

            if abs(x) > ACCEL_LIMIT or abs(y) > ACCEL_LIMIT or abs(z) > ACCEL_LIMIT:
                print(f"Skipping outlier: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                return

            hpf_x = HPF_ALPHA * (prev_hpf_x + x - prev_input_x)
            hpf_y = HPF_ALPHA * (prev_hpf_y + y - prev_input_y)
            hpf_z = HPF_ALPHA * (prev_hpf_z + z - prev_input_z)
            lpf_x = LPF_BETA * hpf_x + (1 - LPF_BETA) * prev_lpf_x
            lpf_y = LPF_BETA * hpf_y + (1 - LPF_BETA) * prev_lpf_y
            lpf_z = LPF_BETA * hpf_z + (1 - LPF_BETA) * prev_lpf_z

            prev_hpf_x = hpf_x
            prev_hpf_y = hpf_y
            prev_hpf_z = hpf_z
            prev_lpf_x = lpf_x
            prev_lpf_y = lpf_y
            prev_lpf_z = lpf_z
            prev_input_x = x
            prev_input_y = y
            prev_input_z = z

            vel_x += ((lpf_x + prev_acc_x) / 2) * DT
            vel_y += ((lpf_y + prev_acc_y) / 2) * DT
            vel_z += ((lpf_z + prev_acc_z) / 2) * DT
            prev_acc_x = lpf_x
            prev_acc_y = lpf_y
            prev_acc_z = lpf_z

            disp_x += ((vel_x + prev_vel_x) / 2) * DT
            disp_y += ((vel_y + prev_vel_y) / 2) * DT
            disp_z += ((vel_z + prev_vel_z) / 2) * DT
            prev_vel_x = vel_x
            prev_vel_y = vel_y
            prev_vel_z = vel_z

            if sample_counter >= RESET_INTERVAL:
                vel_x = vel_y = vel_z = 0
                disp_x = disp_y = disp_z = 0
                sample_counter = 0

            vel_buffer_x[vel_index] = vel_x
            vel_buffer_y[vel_index] = vel_y
            vel_buffer_z[vel_index] = vel_z
            disp_buffer_x[vel_index] = disp_x
            disp_buffer_y[vel_index] = disp_y
            disp_buffer_z[vel_index] = disp_z
            vel_index = (vel_index + 1) % VEL_WINDOW

            vel_rms_x = np.sqrt(sum(v * v for v in vel_buffer_x) / VEL_WINDOW)
            vel_rms_y = np.sqrt(sum(v * v for v in vel_buffer_y) / VEL_WINDOW)
            vel_rms_z = np.sqrt(sum(v * v for v in vel_buffer_z) / VEL_WINDOW)
            disp_rms_x = np.sqrt(sum(v * v for v in disp_buffer_x) / VEL_WINDOW)
            disp_rms_y = np.sqrt(sum(v * v for v in disp_buffer_y) / VEL_WINDOW)
            disp_rms_z = np.sqrt(sum(v * v for v in disp_buffer_z) / VEL_WINDOW)

            vel_rms_h = float(vel_rms_z)  # Horizontal = Z
            vel_rms_v = float(vel_rms_y)  # Vertical = Y
            vel_rms_a = float(vel_rms_x)  # Axial = X
            disp_rms_h = float(disp_rms_z)
            disp_rms_v = float(disp_rms_y)
            disp_rms_a = float(disp_rms_x)

            row = (sample_count, x, y, z, h_vib, v_vib, a_vib, vel_rms_h, vel_rms_v, vel_rms_a, disp_rms_h, disp_rms_v, disp_rms_a, sensor_id)
            all_data.append(row)
            print(f"Added Data: sample={sample_count}, x={x:.2f}, y={y:.2f}, z={z:.2f}, h_vib={h_vib:.2f}, sensor_id={sensor_id}")

            if len(all_data) >= 10:
                try:
                    cursor.executemany("""
                        INSERT INTO samples (sample, x, y, z, h_vib, v_vib, a_vib, h_vel, v_vel, a_vel, h_disp, v_disp, a_disp, sensor_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, all_data)
                    conn.commit()
                    print(f"✅ Saved {len(all_data)} rows to vibration_data_db")
                    all_data.clear()
                except Exception as e:
                    print(f"Failed to save data: {e}")
                    conn.rollback()
                    all_data.clear()

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        conn.rollback()

def on_error(ws, error):
    print(f"WebSocket error: {error}")
    try:
        conn.rollback()
    except Exception as e:
        print(f"Error during rollback: {e}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")
    try:
        if conn:
            if all_data:
                cursor.executemany("""
                    INSERT INTO samples (sample, x, y, z, h_vib, v_vib, a_vib, h_vel, v_vel, a_vel, h_disp, v_disp, a_disp, sensor_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, all_data)
                conn.commit()
                print(f"✅ Saved {len(all_data)} remaining rows to vibration_data_db")
                all_data.clear()
            cursor.close()
            conn.close()
            print("Database connection closed.")
    except Exception as e:
        print(f"Failed to save remaining data or close connection: {e}")

def on_open(ws):
    print("WebSocket connection opened")
    ws.send("Authorization: Basic YXNoOmFzaDEyMw==")
    def ping():
        if ws.sock and ws.sock.connected:
            ws.send("ping")
        threading.Timer(30, ping).start()  # Ping every 30 seconds
    ping()

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://104.34.48.162:8081/",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
