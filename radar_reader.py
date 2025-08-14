# radar_reader.py
import os
import sys
import csv
import time
import serial  
import struct
import signal
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from scipy.signal import spectrogram

# ------------------------- Config -------------------------
SERIAL_PORT   = os.getenv("RADAR_PORT", "/dev/tty.usbserial-DA0151YQ")
SERIAL_BAUD   = 230400

M0_EXPECTED   = 0xA5A5
M1_EXPECTED   = 0x0A0D

FRAME_SIZE    = 532          # bytes: <H 128I 4I H> = 2 + 512 + 16 + 2
WORDS_PER_FR  = 128          # 128 IQ words per frame (each word packs I[15:0], Q[31:16])

FS_RADAR_HZ   = 256.0
SAMPLES_PER_FR= WORDS_PER_FR
FRAME_DUR_S   = SAMPLES_PER_FR / FS_RADAR_HZ  # 0.5 s per frame at 256 Hz

# Timestamp tick (µs per tick). Use the value that matches your firmware clock.
# If your device uses 16.384 kHz → 61.0352 µs/tick; if 32.768 kHz → 30.5176 µs/tick.
TS_TICK_US    = 61.0352

# Beacon debouncing / edge-merge (host-side fix for double-edge counting)
PULSE_WIDTH_S   = 0.250      # High time from Vicon program (250 ms)
PULSE_PERIOD_S  = 0.750      # Pulse period from Vicon program (750 ms)
EDGE_MERGE_MAX_S= 0.35       # If two edges are within ~0.25 s, treat as the same pulse

# ------------------------- CRC32 (STM32 poly) -------------------------
def crc32_wordwise(words):
    """
    Replicates the C-side CRC: init=0xFFFFFFFF, poly=0x04C11DB7, 32-bit word-by-word.
    """
    poly = 0x04C11DB7
    crc  = 0xFFFFFFFF
    for w in words:
        crc ^= (w & 0xFFFFFFFF)
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc

# ------------------------- Serial helpers -------------------------
def read_exactly(ser: serial.Serial, n: int) -> bytes:
    """Blocking read of exactly n bytes (raises on timeout)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("Serial read timeout")
        buf.extend(chunk)
    return bytes(buf)

def align_to_header(ser: serial.Serial) -> bytes:
    """
    Align to a frame that starts with 0xA5 0xA5. Returns the full 532-byte frame.
    We consume the two 0xA5 bytes, then read the remaining bytes of the frame.
    """
    dropped = 0
    saw_a5  = False
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("Serial read timeout while aligning")
        if b[0] == 0xA5:
            if saw_a5:
                # Found 0xA5 0xA5
                break
            saw_a5 = True
        else:
            saw_a5 = False
            dropped += 1
    if dropped:
        print(f"[align] dropped {dropped} stray bytes before header")

    rest = read_exactly(ser, FRAME_SIZE - 2)  # already consumed 2 bytes
    return b"\xA5\xA5" + rest

def open_serial(port: str, baud: int) -> serial.Serial:
    """Open serial port in raw 8N1."""
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
        write_timeout=1,
    )
    return ser


def save_doppler_spectrogram_png(I_list, Q_list, fs_hz, out_png,
                                 nperseg=256, noverlap=224):
    
    
    if len(I_list) == 0:
        print("[warn] No IQ samples captured; skip spectrogram export.")
        return
    
    I = np.asarray(I_list, dtype=np.uint16).view(np.int16).astype(np.float32)
    Q = np.asarray(Q_list, dtype=np.uint16).view(np.int16).astype(np.float32)
    x = I + 1j * Q

    frequencies, times, Sxx = spectrogram(
    x,
    fs=fs_hz,
    window='hann',
    nperseg = nperseg,
    noverlap= noverlap,
    detrend='constant',
    return_onesided=False,
    scaling='density',
    mode='complex'
    )
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    Sxx_magnitude = np.abs(Sxx_shifted)
    Sxx_sqrt = np.sqrt(Sxx_magnitude)

    frequencies_shifted = np.fft.fftshift(frequencies)
    # 6) 畫圖與存檔
    plt.figure(figsize=(10, 4), dpi=150)
    plt.pcolormesh(times, frequencies_shifted, Sxx_sqrt, shading='gouraud', cmap='magma')
    plt.xlabel("Time (s)")
    plt.ylabel("Doppler (Hz)")
    plt.title("Radar Doppler Spectrogram (STFT)")
    cbar = plt.colorbar()
    cbar.set_label("Amplitude (sqrt scale)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"[info] Saved spectrogram PNG -> {out_png}")
    

def compute_beacon_sample_offset(radar_ts, beacon_ts, fs_hz, ts_tick_us):

    ticks_per_sample = (1.0 / fs_hz) / (ts_tick_us / 1_000_000.0)
    offset_float = (beacon_ts - radar_ts) / ticks_per_sample
    offset_int = int(round(offset_float))
    return offset_int, offset_float, ticks_per_sample


# ------------------------- Main -------------------------
def main():

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    all_I_u16 = []
    all_Q_u16 = []
    stop_flag = {"stop": False} 
    start_monotonic = time.monotonic()
    def handle_sig(sig, frame):
        # 不直接 sys.exit；只設旗標，讓 loop 優雅收尾
        stop_flag["stop"] = True
        print("\n[info] Stop requested (signal). Finishing current iteration...")

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    synced = (not args.sync_on_beacon)   # 若沒開此功能，一開始就視為已同步
    t0_beacon_time_s = None              # 第一個 beacon 的實際時間（radar 時基）
    global_sample_idx = 0                # 對齊後的樣本序號（從 0 開始）
    FRAME_BUFFER_SECS = 10               # 緩衝 10 秒（可再調）
    BUF_MAX_FRAMES = int(FRAME_BUFFER_SECS / FRAME_DUR_S)
    buf_frames = deque(maxlen=BUF_MAX_FRAMES)
    if args.hotkey:
        def _hotkey_listener():
            for line in sys.stdin:
                if line.strip().lower() == "q":
                    stop_flag["stop"] = True
                    print("[info] Stop requested (hotkey 'q').")
                    break
        threading.Thread(target=_hotkey_listener, daemon=True).start()

    # Output file names
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    iq_fname   = os.path.join(args.outdir, f"{timestamp}_iq.csv")
    meta_fname = os.path.join(args.outdir, f"{timestamp}_frame_meta.csv")

    print(f"[info] Writing IQ CSV   -> {iq_fname}")
    print(f"[info] Writing META CSV -> {meta_fname}")

    f_iq   = open(iq_fname, "w", newline="")
    f_meta = open(meta_fname, "w", newline="")
    iq_writer   = csv.writer(f_iq)
    meta_writer = csv.writer(f_meta)

    # CSV headers
    # One row per sample (128 rows per frame):
    # beacon_sample_offset: 相對於本 frame 第 0 筆的偏移（整數）
    # is_beacon_sample: 當前 sample 是否等於該偏移
    iq_writer.writerow([
        "frame_idx","sample_idx","global_sample_idx","global_time_s",
        "beacon_count_raw","beacon_pulse_count","I_u16","Q_u16",
        "beacon_sample_offset","is_beacon_sample"
    ])
    # One row per frame:

    meta_writer.writerow([
        "frame_idx","radar_ts","beacon_ts","beacon_count_raw",
        "beacon_time_s","radar_time_s","beacon_pulse_count",
        "beacon_sample_offset","beacon_in_frame","ticks_per_sample","offset_float",
        "t0_beacon_time_s","synced"
    ])

    # Graceful Ctrl+C
    #def _cleanup(sig, frame):
    #    print("\n[info] Caught SIGINT, flushing & closing...")
    #    try:
    #        f_iq.flush();  f_iq.close()
    #        f_meta.flush(); f_meta.close()
    #    finally:
    #        sys.exit(0)
    #signal.signal(signal.SIGINT, _cleanup)
    #signal.signal(signal.SIGINT, handle_sig)
    #signal.signal(signal.SIGTERM, handle_sig)
    # Open serial
    ser = open_serial(SERIAL_PORT, SERIAL_BAUD)
    print(f"[info] Opened {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    last_data_time = time.monotonic()
    # Debounce state
    prev_beacon_ts = None          # last seen raw beacon_ts (uint32)
    last_pulse_time_s = None       # last ACCEPTED pulse (rising-edge surrogate) time in seconds
    beacon_pulse_count = 0         # debounced pulse counter (host-side)

    frame_idx = 0

    try:
        while True:
            # 1) Align and read one full frame
            #frame = align_to_header(ser)
            try:
                frame = align_to_header(ser)
                last_data_time = time.monotonic()  
            except TimeoutError:
                if args.idle is not None and (time.monotonic() - last_data_time) >= args.idle:
                    print("[warn] Idle timeout reached while aligning; stopping.")
                    break 
                continue
            # 2) Unpack: <H 128I 4I H>
            #    m0, payload[128], radar_ts, beacon_ts, beacon_count, crc32, m1
            unpacked = struct.unpack("<H128I4IH", frame)
            m0 = unpacked[0]
            payload_words = unpacked[1:1+WORDS_PER_FR]
            radar_ts, beacon_ts, beacon_count_raw, rx_crc32 = unpacked[1+WORDS_PER_FR:1+WORDS_PER_FR+4]
            m1 = unpacked[-1]

            # 3) Sanity checks
            if m0 != M0_EXPECTED or m1 != M1_EXPECTED:
                print(f"[warn] marker mismatch m0=0x{m0:04X}, m1=0x{m1:04X}")
                continue

            local_crc = crc32_wordwise(list(payload_words) + [radar_ts, beacon_ts, beacon_count_raw])
            if local_crc != rx_crc32:
                print(f"[warn] CRC FAIL rx=0x{rx_crc32:08X} local=0x{local_crc:08X}")
               

            # 4) Convert timestamps (ticks -> seconds)
            radar_time_s  = radar_ts  * (TS_TICK_US / 1_000_000.0)
            beacon_time_s = beacon_ts * (TS_TICK_US / 1_000_000.0)
            offset_int, offset_float, ticks_per_sample = compute_beacon_sample_offset(
                radar_ts, beacon_ts, FS_RADAR_HZ, TS_TICK_US
            )
            beacon_in_frame = 1 if 0 <= offset_int < WORDS_PER_FR else 0

            if beacon_count_raw == 0 or beacon_time_s == 0.0:
                offset_int_for_csv = ""
                beacon_in_frame = 0
            else:
                offset_int_for_csv = offset_int
                beacon_in_frame = 1 if 0 <= offset_int < WORDS_PER_FR else 0
            # 5) Host-side beacon debouncing / edge-merge:
            #    We only consider a NEW edge when beacon_ts changes.
            #    If the interval from the last ACCEPTED pulse is ~0.25 s, ignore (likely the opposite edge of the same pulse).
            #    Otherwise (~0.5 s or more), accept and increment beacon_pulse_count.
            if prev_beacon_ts is None or beacon_ts != prev_beacon_ts:
                if last_pulse_time_s is None:
                    if beacon_time_s > 0:          
                        beacon_pulse_count += 1
                        last_pulse_time_s = beacon_time_s
                else:
                    dt = beacon_time_s - last_pulse_time_s
                    if dt > EDGE_MERGE_MAX_S:
                        beacon_pulse_count += 1
                        last_pulse_time_s = beacon_time_s
                prev_beacon_ts = beacon_ts

            # 6) Unpack I/Q as unsigned 16-bit 
            I_vals = [(w & 0xFFFF) for w in payload_words]
            Q_vals = [((w >> 16) & 0xFFFF) for w in payload_words]

            frame_meta = {
                "frame_idx": frame_idx,
                "radar_ts": radar_ts,
                "beacon_ts": beacon_ts,
                "beacon_count_raw": beacon_count_raw,
                "beacon_time_s": beacon_time_s,
                "radar_time_s": radar_time_s,
                "beacon_pulse_count": beacon_pulse_count,
                "offset_int": (offset_int if beacon_in_frame else None),
                "offset_int_for_csv": offset_int_for_csv,
                "beacon_in_frame": beacon_in_frame,
                "ticks_per_sample": ticks_per_sample,
                "offset_float": offset_float,
            }

            if not synced:
                # 先把 frame 放進緩衝（不落盤）
                buf_frames.append((I_vals, Q_vals, frame_meta))

                # 條件：只要偵測到本 frame 有 beacon（或 beacon_ts 有更新且>0），就觸發同步
                first_edge_seen = (beacon_in_frame == 1) or ((prev_beacon_ts is not None) and (beacon_ts != prev_beacon_ts) and (beacon_time_s > 0))
                if first_edge_seen:
                    synced = True
                    t0_beacon_time_s = beacon_time_s
                    # 從觸發這一幀的「beacon sample」開始寫入；之前的幀全部丟棄（如要保留前幾秒可擴充）
                    I_last, Q_last, M_last = buf_frames[-1]
                    start_i = M_last["offset_int"] if M_last["offset_int"] is not None else 0

                    # 從對齊 sample 開始寫 IQ
                    for i in range(start_i, len(I_last)):
                        t_s = global_sample_idx / FS_RADAR_HZ
                        is_beacon_sample = 1 if (i == start_i) else 0
                        iq_writer.writerow([
                            M_last["frame_idx"], i, global_sample_idx, f"{t_s:.6f}",
                            M_last["beacon_count_raw"], M_last["beacon_pulse_count"],
                            I_last[i], Q_last[i],
                            (M_last["offset_int"] if M_last["offset_int"] is not None else ""),
                            is_beacon_sample
                        ])
                        # 對齊後才開始累積到頻譜資料
                        all_I_u16.append(I_last[i]); all_Q_u16.append(Q_last[i])
                        global_sample_idx += 1

                    # 寫出這幀的 META（標註 synced）
                    meta_writer.writerow([
                        M_last["frame_idx"], M_last["radar_ts"], M_last["beacon_ts"], M_last["beacon_count_raw"],
                        f"{M_last['beacon_time_s']:.6f}", f"{M_last['radar_time_s']:.6f}", M_last["beacon_pulse_count"],
                        (M_last["offset_int"] if M_last["offset_int"] is not None else ""),
                        M_last["beacon_in_frame"], f"{M_last['ticks_per_sample']:.3f}", f"{M_last['offset_float']:.3f}",
                        f"{t0_beacon_time_s:.6f}", 1
                    ])

                    # 觸發後清空緩衝（或改成保留以後幀）
                    buf_frames.clear()
                    f_iq.flush(); f_meta.flush()

            else:
                # 已同步：正常把整幀寫入（時間以 beacon 為 0）
                for i, (I_u16, Q_u16) in enumerate(zip(I_vals, Q_vals)):
                    t_s = global_sample_idx / FS_RADAR_HZ
                    is_beacon_sample = 1 if (beacon_in_frame and (offset_int == i)) else 0
                    iq_writer.writerow([
                        frame_idx, i, global_sample_idx, f"{t_s:.6f}",
                        beacon_count_raw, beacon_pulse_count,
                        I_u16, Q_u16, offset_int_for_csv, is_beacon_sample
                    ])
                    all_I_u16.append(I_u16); all_Q_u16.append(Q_u16)
                    global_sample_idx += 1

                meta_writer.writerow([
                    frame_idx, radar_ts, beacon_ts, beacon_count_raw,
                    f"{beacon_time_s:.6f}", f"{radar_time_s:.6f}", beacon_pulse_count,
                    offset_int_for_csv, beacon_in_frame, f"{ticks_per_sample:.3f}", f"{offset_float:.3f}",
                    (f"{t0_beacon_time_s:.6f}" if t0_beacon_time_s is not None else ""), 1
                ])
                f_iq.flush(); f_meta.flush()

            # brief status print
            off_str = (str(offset_int) if (beacon_count_raw > 0 and beacon_in_frame) else "n/a")
            print(
                f"frame {frame_idx:6d} | raw_count={beacon_count_raw:6d} | pulse_count={beacon_pulse_count:6d} "
                f"| radar_t={radar_time_s:9.3f}s | beacon_t={beacon_time_s:9.3f}s "
                f"| offset={off_str:>4} ({'in' if beacon_in_frame else 'out'})"
                )
            if args.duration is not None:
                if (time.monotonic() - start_monotonic) >= args.duration:
                    print("[info] Duration reached; stopping.")
                    break
            if args.frames is not None and (frame_idx + 1) >= args.frames:
                print("[info] Frame limit reached; stopping.")
                break
            if args.pulses is not None and beacon_pulse_count >= args.pulses:
                print("[info] Pulse count reached; stopping.")
                break
            if args.idle is not None and (time.monotonic() - last_data_time) >= args.idle:
                print("[warn] Idle timeout reached; stopping.")
                break
            if stop_flag["stop"]:
                print("[info] Stop flag set; stopping.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
         pass
    finally:
        try:
            f_iq.close(); f_meta.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        if args.sync_on_beacon and global_sample_idx == 0:
            print("[warn] No beacon detected during run; no aligned samples were saved.")
        try:
            png_path = os.path.join(args.outdir, f"{timestamp}_spectrogram.png")
            save_doppler_spectrogram_png(all_I_u16, all_Q_u16, FS_RADAR_HZ, png_path,
                                         nperseg=256, noverlap=224)
        except Exception as e:
            print(f"[error] Failed to save spectrogram: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=None, help="Record N seconds then stop")
    p.add_argument("--frames", type=int, default=None, help="Record N frames then stop")
    p.add_argument("--pulses", type=int, default=None, help="Stop after N debounced beacon pulses")
    p.add_argument("--idle", type=float, default=None, help="Stop if no bytes arrive for N seconds")
    p.add_argument("--hotkey", action="store_true", help="Enable 'q'+Enter to stop")
    p.add_argument("--outdir", type=str, default="radar", help="Output directory for CSV/PNG files")
    p.add_argument("--sync-on-beacon", action="store_true",
               help="Wait for the first beacon edge, then start saving aligned at that edge (t=0)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"[error] Serial error: {e}")
        sys.exit(1)
