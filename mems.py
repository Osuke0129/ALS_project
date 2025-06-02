import serial
import time
import msvcrt
import csv
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r'C:\Users\217ki\ALS_project\result\MEMS_result'
BAUD_RATE = 115200
SERIAL_PORT = 'COM4'
current_mode = 'r'

def parse_sensor_data(line):
    try:
        parts = line.strip().split(',')
        if len(parts) == 3:
            return [float(p) for p in parts]
    except ValueError:
        pass
    return None

def plot_graph(csv_path, output_image_path, mode_label):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['elapsed_sec'] = (df['timestamp'] - df['timestamp'][0]).dt.total_seconds()

    plt.figure(figsize=(12, 6))
    plt.plot(df['elapsed_sec'], df['x'], label='X', linewidth=1)
    plt.plot(df['elapsed_sec'], df['y'], label='Y', linewidth=1)
    plt.plot(df['elapsed_sec'], df['z'], label='Z', linewidth=1)

    plt.title(f"Sensor Data Graph (Mode: {mode_label})")
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_image_path)
    print(f"→ グラフ画像を保存しました: {output_image_path}")
    plt.show()

def record_once(save_dir, trial_num):
    global current_mode

    trial_folder = os.path.join(save_dir, f"trial_{trial_num}")
    os.makedirs(trial_folder, exist_ok=True)
    csv_path = os.path.join(trial_folder, "sensor_log.csv")
    img_path = os.path.join(trial_folder, f"trial_{trial_num}.png")

    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'mode', 'x', 'y', 'z'])

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.write(b'r')
        print(f"\n{trial_num}回目の記録を開始します(Ctrl+Cで終了)")

        while True:
            line = ser.readline().decode('utf-8').strip()
            data = parse_sensor_data(line)
            if data:
                x, y, z = data[2], data[1], data[0]
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                if current_mode == 'r':
                    print(f"[RAW]   X={x:.0f}, Y={y:.0f}, Z={z:.0f}")
                #elif current_mode == 'n':
                    #print(f"[FORCE] X={x:.3f}N, Y={y:.3f}N, Z={z:.3f}N")

                csv_writer.writerow([now_str, current_mode, x, y, z])

            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key in ('r', 'n', 'o'):
                    ser.write(key.encode())
                    current_mode = key
                    print(f"\n→ モード切替: '{key}' を送信しました。\n")

    except KeyboardInterrupt:
        print(f"\n {trial_num}回目の記録を終了しました。")
    except serial.SerialException:
        print("シリアルポートが見つかりません。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        csv_file.close()
        print(f"CSVを保存しました: {csv_path}")
        print(f"グラフ描画中...")
        plot_graph(csv_path, img_path, current_mode)

def main():
    user_name = input("名前を入力してください（半角英数字）: ").strip().replace(' ', '_')
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    session_name = f"{user_name}_{timestamp_str}"
    save_dir = os.path.join(BASE_DIR, session_name)
    os.makedirs(save_dir, exist_ok=True)

    trial_count = 1

    while True:
        q = input(f"\n記録しますか?(y/n)→ {trial_count}回目 : ").strip().lower()
        if q != 'y':
            print("終了します。")
            break

        start = input(f"{trial_count}回目を開始してよいですか?(y/n): ").strip().lower()
        if start == 'y':
            record_once(save_dir, trial_count)
            trial_count += 1
        else:
            print("この回数はスキップされました。")

if __name__ == "__main__":
    main()
