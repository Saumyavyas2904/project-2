import os
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import datetime
import time
from scipy.interpolate import make_interp_spline
from psycopg2 import connect, Error
from flask import Flask, render_template_string, request
import logging
import warnings

# Suppress Python 3.13 date parsing warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set modern font
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# Initialize Flask app
app = Flask(__name__)

# Embedded HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vibration Data Dashboard</title>
    <meta http-equiv="refresh" content="0.5" id="refresh">
    <style>
        body { background-color: #f5f5f5; font-family: Arial, sans-serif; margin: 20px; }
        img { max-width: 100%; height: auto; }
        .container { max-width: 1000px; margin: auto; }
        button { margin: 10px; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vibration Data Dashboard</h1>
        <form method="post">
            <button type="submit" name="pause">{{ 'Resume' if is_paused else 'Pause' }}</button>
        </form>
        <img src="data:image/png;base64,{{ img_data }}" alt="Vibration Plot">
    </div>
    <script>
        var refresh = document.getElementById('refresh');
        if ({{ is_paused|tojson }}) {
            refresh.remove();
        }
    </script>
</body>
</html>
"""

# Initialize lists and state
time_values = []
h_vel_values = []
v_vel_values = []
a_vel_values = []
h_freq_values = []
v_freq_values = []
a_freq_values = []
all_time_values = []
all_h_vel_values = []
all_v_vel_values = []
all_a_vel_values = []
current_sensor_id = {'value': 1}
is_valid_sensor_id = {'value': True}
selected_month = {'value': None}
is_month_active = {'value': False}
time_window = 120
freq_window = 10
start_time = time.time()
is_paused = {'value': False}
window_start_time = {'value': None}
scroll_step = 60
is_scrolling = {'value': True}
show_h = {'value': True}
show_v = {'value': True}
show_a = {'value': True}

# Database connection with retry logic
def connect_db():
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            conn = connect(
                dbname=os.getenv("DB_NAME", "vibration_data_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "123"),
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432")
            )
            cursor = conn.cursor()
            logger.info("Connected to PostgreSQL successfully!")
            return conn, cursor
        except Error as e:
            logger.error(f"Error connecting to PostgreSQL (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Failed to connect to database after retries. Exiting.")
                exit()
conn, cursor = connect_db()

def fetch_latest_data():
    try:
        logger.info("Fetching data...")
        if not is_valid_sensor_id['value'] or current_sensor_id['value'] is None:
            logger.warning("No valid sensor ID.")
            return None, None, None, None
            
        if is_month_active['value'] and selected_month['value'] is not None:
            current_year = datetime.datetime.now().year
            start_date = datetime.datetime(current_year, selected_month['value'], 1)
            end_date = (start_date + datetime.timedelta(days=31)).replace(day=1, year=current_year) - datetime.timedelta(days=1)
            logger.info(f"Querying data for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            cursor.execute("""
                SELECT h_vel, v_vel, a_vel, timestamp
                FROM samples 
                WHERE sensor_id = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """, (current_sensor_id['value'], start_date, end_date))
            rows = cursor.fetchall()
            if rows:
                times = []
                h_vels = []
                v_vels = []
                a_vels = []
                for row in rows:
                    h_vel = float(row[0]) if row[0] is not None else 0.0
                    v_vel = float(row[1]) if row[1] is not None else 0.0
                    a_vel = float(row[2]) if row[2] is not None else 0.0
                    timestamp = row[3]
                    if timestamp is not None:
                        times.append(timestamp)
                        h_vels.append(h_vel)
                        v_vels.append(v_vel)
                        a_vels.append(a_vel)
                    else:
                        logger.warning("Skipping row with None timestamp")
                logger.info(f"Fetched {len(times)} rows.")
                return times, h_vels, v_vels, a_vels
            logger.info("No data found for the selected month.")
            return None, None, None, None
            
        cursor.execute("""
            SELECT h_vel, v_vel, a_vel, EXTRACT(EPOCH FROM timestamp) AS epoch_time 
            FROM samples 
            WHERE sensor_id = %s 
            ORDER BY timestamp DESC LIMIT 1
        """, (current_sensor_id['value'],))
        row = cursor.fetchone()
        if row:
            h_vel = float(row[0]) if row[0] is not None else 0.0
            v_vel = float(row[1]) if row[1] is not None else 0.0
            a_vel = float(row[2]) if row[2] is not None else 0.0
            timestamp = float(row[3]) if row[3] is not None else time.time()
            current_time = timestamp - start_time
            logger.info(f"Fetched latest data: time={current_time}, h_vel={h_vel}, v_vel={v_vel}, a_vel={a_vel}")
            return current_time, h_vel, v_vel, a_vel
        logger.info("No data found for the latest point.")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None, None, None, None

def compute_real_frequency(times, velocities):
    if len(times) < 2 or len(velocities) < 2:
        return 0.01
    if isinstance(times[0], datetime.datetime):
        times = [pd.Timestamp(t).timestamp() for t in times]
    current_time = times[-1]
    window_start = current_time - freq_window
    window_times = []
    window_velocities = []
    for t, v in zip(times, velocities):
        if t >= window_start:
            window_times.append(t)
            window_velocities.append(v)
    if len(window_times) < 2 or len(window_velocities) < 2:
        return 0.01
    vel_mean = np.mean(window_velocities)
    centered_vel = np.array(window_velocities) - vel_mean
    zero_crossings = 0
    for i in range(1, len(centered_vel)):
        if centered_vel[i-1] * centered_vel[i] < 0:
            zero_crossings += 1
    time_duration = window_times[-1] - window_times[0]
    if time_duration <= 0:
        return 0.01
    freq = (zero_crossings / 2) / time_duration
    return max(0.01, min(freq, 10))

def update_xticks_with_military_time(ax, ticks, start_time):
    labels = []
    for tick in ticks:
        abs_time = start_time + tick
        dt = datetime.datetime.fromtimestamp(abs_time, tz=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        military_time = dt.strftime('%H:%M')
        labels.append(f"{int(tick)}\n{military_time}")
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

def generate_plot():
    global time_values, h_vel_values, v_vel_values, a_vel_values
    global h_freq_values, v_freq_values, a_freq_values
    global all_time_values, all_h_vel_values, all_v_vel_values, all_a_vel_values
    global window_start_time

    if not is_paused['value']:
        current_time = time.time() - start_time
        data = fetch_latest_data()
        
        if data[0] is not None:
            if is_month_active['value']:
                times, h_vels, v_vels, a_vels = data
                if times:
                    logger.info("Updating with month data...")
                    time_values.clear()
                    h_vel_values.clear()
                    v_vel_values.clear()
                    a_vel_values.clear()
                    h_freq_values.clear()
                    v_freq_values.clear()
                    a_freq_values.clear()
                    all_time_values.clear()
                    all_h_vel_values.clear()
                    all_v_vel_values.clear()
                    all_a_vel_values.clear()
                    
                    for t, h, v, a in zip(times, h_vels, v_vels, a_vels):
                        time_values.append(t)
                        h_vel_values.append(h)
                        v_vel_values.append(v)
                        a_vel_values.append(a)
                        h_freq = compute_real_frequency(time_values, h_vel_values)
                        v_freq = compute_real_frequency(time_values, v_vel_values)
                        a_freq = compute_real_frequency(time_values, a_vel_values)
                        h_freq_values.append(h_freq)
                        v_freq_values.append(v_freq)
                        a_freq_values.append(a_freq)
                        all_time_values.append(t)
                        all_h_vel_values.append(h)
                        all_v_vel_values.append(v)
                        all_a_vel_values.append(a)
            else:
                time_val, h_vel, v_vel, a_vel = data
                time_values.append(time_val)
                h_vel_values.append(h_vel)
                v_vel_values.append(v_vel)
                a_vel_values.append(a_vel)
                h_freq = compute_real_frequency(time_values, h_vel_values)
                v_freq = compute_real_frequency(time_values, v_vel_values)
                a_freq = compute_real_frequency(time_values, a_vel_values)
                h_freq_values.append(h_freq)
                v_freq_values.append(v_freq)
                a_freq_values.append(a_freq)
                all_time_values.append(time_val)
                all_h_vel_values.append(h_vel)
                all_v_vel_values.append(v_vel)
                all_a_vel_values.append(a_vel)

        if not is_month_active['value']:
            while time_values and current_time - time_values[0] > time_window:
                time_values.pop(0)
                h_vel_values.pop(0)
                v_vel_values.pop(0)
                a_vel_values.pop(0)
                h_freq_values.pop(0)
                v_freq_values.pop(0)
                a_freq_values.pop(0)

    if window_start_time['value'] is None:
        window_start_time['value'] = max(0, current_time - time_window)
    if is_scrolling['value'] and not is_month_active['value']:
        window_start_time['value'] = max(0, current_time - time_window)
    window_start = window_start_time['value']
    window_end = window_start + time_window

    if is_month_active['value'] and selected_month['value'] is not None:
        current_year = datetime.datetime.now().year
        start_date = datetime.datetime(current_year, selected_month['value'], 1)
        end_date = (start_date + datetime.timedelta(days=31)).replace(day=1, year=current_year) - datetime.timedelta(days=1)
        window_start = pd.Timestamp(start_date).timestamp() - start_time
        window_end = pd.Timestamp(end_date).timestamp() - start_time

    display_times = []
    display_h_vel = []
    display_v_vel = []
    display_a_vel = []
    for t, h, v, a in zip(all_time_values, all_h_vel_values, all_v_vel_values, all_a_vel_values):
        t_num = t if not isinstance(t, datetime.datetime) else pd.Timestamp(t).timestamp() - start_time
        if window_start <= t_num <= window_end:
            display_times.append(t_num)
            display_h_vel.append(h)
            display_v_vel.append(v)
            display_a_vel.append(a)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    fig.patch.set_facecolor('#f5f5f5')
    plt.subplots_adjust(top=0.92, bottom=0.2, hspace=0.3, left=0.1, right=0.9)

    # Velocity vs Time
    if len(display_times) > 3:
        data = list(zip(display_times, display_h_vel, display_v_vel, display_a_vel))
        data.sort(key=lambda x: x[0])
        seen_times = {}
        for t, h, v, a in data:
            seen_times[t] = (h, v, a)
        display_times = list(seen_times.keys())
        display_h_vel = [x[0] for x in seen_times.values()]
        display_v_vel = [x[1] for x in seen_times.values()]
        display_a_vel = [x[2] for x in seen_times.values()]
        display_times_np = np.array(display_times)
        display_h_vel_np = np.array(display_h_vel)
        display_v_vel_np = np.array(display_v_vel)
        display_a_vel_np = np.array(display_a_vel)
        num_points = len(display_times) * 5
        smooth_times = np.linspace(display_times_np.min(), display_times_np.max(), num_points)
        h_spline = make_interp_spline(display_times_np, display_h_vel_np, k=3)
        v_spline = make_interp_spline(display_times_np, display_v_vel_np, k=3)
        a_spline = make_interp_spline(display_times_np, display_a_vel_np, k=3)
        smooth_h_vel = h_spline(smooth_times)
        smooth_v_vel = v_spline(smooth_times)
        smooth_a_vel = a_spline(smooth_times)
        x_time = smooth_times
        h_y = smooth_h_vel
        v_y = smooth_v_vel
        a_y = smooth_a_vel
    else:
        x_time = display_times
        h_y = display_h_vel
        v_y = display_v_vel
        a_y = display_a_vel

    if show_h['value']:
        ax1.plot(x_time, h_y, color='#1f77b4', linewidth=1, label='H Vel (mm/s)')
    if show_v['value']:
        ax1.plot(x_time, v_y, color='#ff7f0e', linewidth=1, label='V Vel (mm/s)')
    if show_a['value']:
        ax1.plot(x_time, a_y, color='#2ca02c', linewidth=1, label='A Vel (mm/s)')

    slider_x = window_start + (window_end - window_start) / 2
    t_vals = [t if not isinstance(t, datetime.datetime) else pd.Timestamp(t).timestamp() - start_time for t in all_time_values]
    if t_vals:
        index = min(range(len(t_vals)), key=lambda i: abs(t_vals[i] - slider_x), default=0)
        time_str = f"{t_vals[index]:.2f}s"
        slider_annot_text = f"Time: {time_str}\nH Vel: {all_h_vel_values[index]:.2f} mm/s\nV Vel: {all_v_vel_values[index]:.2f} mm/s\nA Vel: {all_a_vel_values[index]:.2f} mm/s"
        ax1.axvline(slider_x, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.text(slider_x, 25, slider_annot_text, fontsize=9, bbox=dict(facecolor='#e6f3fa', edgecolor='gray', alpha=0.9))
    ax1.set_title("Velocity vs Time", fontsize=12, weight='bold', pad=10)
    ax1.set_xlabel("Time", fontsize=10)
    ax1.set_ylabel("Velocity (mm/s)", fontsize=10)
    ax1.set_ylim(0, 25)
    ax1.set_xlim(window_start, window_end)
    ax1.set_xticks(np.arange(max(0, window_start), window_end + 60, 60))
    update_xticks_with_military_time(ax1, np.arange(max(0, window_start), window_end + 60, 60), start_time)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, framealpha=0.8, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax1.set_facecolor('#fafafa')

    # Velocity vs Frequency
    total_y_range = 25
    section_height = total_y_range / 3
    h_base = 0
    v_base = section_height
    a_base = 2 * section_height
    if show_h['value'] and h_freq_values:
        for freq, v in zip(h_freq_values, h_vel_values):
            ax2.plot([freq, freq], [h_base, h_base + v], color='#1f77b4', linewidth=0.5)
    if show_v['value'] and v_freq_values:
        for freq, v in zip(v_freq_values, v_vel_values):
            ax2.plot([freq, freq], [v_base, v_base + v], color='#ff7f0e', linewidth=0.5)
    if show_a['value'] and a_freq_values:
        for freq, v in zip(a_freq_values, a_vel_values):
            ax2.plot([freq, freq], [a_base, a_base + v], color='#2ca02c', linewidth=0.5)
    avg_freq = (h_freq_values[-1] + v_freq_values[-1] + a_freq_values[-1]) / 3 if h_freq_values else 0.01
    if h_freq_values:
        avg_freqs = [(f_h + f_v + f_a) / 3 for f_h, f_v, f_a in zip(h_freq_values, v_freq_values, a_freq_values)]
        index = min(range(len(avg_freqs)), key=lambda i: abs(avg_freqs[i] - avg_freq), default=0)
        if (0 <= index < len(h_vel_values) and 
            0 <= index < len(v_vel_values) and 
            0 <= index < len(a_vel_values)):
            freq_annot_text = f"Freq: {avg_freqs[index]:.2f} Hz\nH Vel: {h_vel_values[index]:.2f} mm/s\nV Vel: {v_vel_values[index]:.2f} mm/s\nA Vel: {a_vel_values[index]:.2f} mm/s"
            ax2.axvline(avg_freq, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax2.text(avg_freq, total_y_range/2, freq_annot_text, fontsize=9, bbox=dict(facecolor='#e6f3fa', edgecolor='gray', alpha=0.9))
    ax2.set_title("Velocity vs Frequency (Real Frequency)", fontsize=12, weight='bold', pad=10)
    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.set_ylabel("Velocity (mm/s)", fontsize=10)
    ax2.set_xscale('log')
    ax2.set_xlim(0.01, 10)
    ax2.set_ylim(0, total_y_range)
    ax2.set_yticks([h_base, h_base + section_height/2, h_base + section_height,
                    v_base, v_base + section_height/2, v_base + section_height,
                    a_base, a_base + section_height/2, a_base + section_height])
    ax2.set_yticklabels(['0', '4', f'{section_height:.1f}', '0', f'{section_height/2:.1f}', f'{section_height:.1f}', '0', f'{section_height/2:.1f}', f'{section_height:.1f}'])
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax2.text(-0.1, h_base + section_height/2, "H Vel (mm/s)", rotation=90, va='center', fontsize=9, transform=ax2.get_yaxis_transform())
    ax2.text(-0.1, v_base + section_height/2, "V Vel (mm/s)", rotation=90, va='center', fontsize=9, transform=ax2.get_yaxis_transform())
    ax2.text(-0.1, a_base + section_height/2, "A Vel (mm/s)", rotation=90, va='center', fontsize=9, transform=ax2.get_yaxis_transform())
    ax2.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax2.set_facecolor('#fafafa')

    # Convert plot to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    global is_paused
    if request.method == 'POST':
        if 'pause' in request.form:
            is_paused['value'] = not is_paused['value']
    img_base64 = generate_plot()
    return render_template_string(HTML_TEMPLATE, img_data=img_base64, is_paused=is_paused['value'])

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
    except KeyboardInterrupt:
        logger.info("Script terminated by user.")
    finally:
        try:
            cursor.close()
            conn.close()
            logger.info("Database connection closed.")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")