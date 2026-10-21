# ===================================================================================
# tc100_shell.py
# Interactive resident CLI controller for 2x TC-100 (X/Y)
#
# Requires: tc100.py (ModbusASCII, Tc100Axis)
# pip install pyserial

import time
import shlex
import math 
from tc100 import ModbusASCII, Tc100Axis

# ======== your setup ========
X_COM, X_ID = "COM3", 2
Y_COM, Y_ID = "COM4", 1
BAUD, PARITY = 19200, "N"
# ===========================


HELP = r"""
Commands (short):

    help                           show this help
    exit | quit | q                exit program

    st                             show status (pos, action, servo)
    pos                            show encoder positions only

    s <n>                           set speed percent (0~100), affects next moves
    son [x|y|xy]                    servo on  (default xy)
    soff [x|y|xy]                   servo off (default xy)
    rst [x|y|xy]                    alarm reset (default xy)

    home [x|y|xy]                   homing (default xy)

    mx <mm>                         move X absolute to mm
    my <mm>                         move Y absolute to mm
    mxy <x_mm> <y_mm>               move X and Y absolute

    dx <mm>                         move X relative by mm
    dy <mm>                         move Y relative by mm
    dxy <dx_mm> <dy_mm>             move X/Y relative

    decel [x|y|xy]                  decel stop (MOV_TYPE=8)
    emg [x|y|xy]                    emergency stop (MOV_TYPE=9)  (use only when necessary)

    scan2d <step_mm> <maxx> <maxy> [dwell_s] [serp=0|1]
                                    run 2D scan from (0,0) to (maxx,maxy) with step
                                    Example: scan2d 2 300 300 0.05 
                                    
    autoscan <t_sec> [x0] [y0]     run 5x5 THz scan, 3cm steps. 
                                    t_sec: pause time in seconds.
                                    If x0, y0 omitted, uses current pos as start.
                                    Example: autoscan 2.5 (uses current pos)
                                    Example: autoscan 2.5 150.0 50.0
                                    
    stitchscan <dwell_s> <n_um> <overlap_pct> [x0] [y0]
                                    run stitching scan for 30x30mm area.
                                    dwell_s: pause time in seconds.
                                    n_um: pixel pitch in um (e.g., 3).
                                    overlap_pct: overlap percentage (e.g., 20).
                                    If x0, y0 omitted, uses current pos as start.
                                    Example: stitchscan 2.5 3 20

Notes:
- All moves are blocking (wait=True) by default in this shell.
- Ctrl+C will try to decel-stop both axes and return to prompt.
""".strip()


def pick_axes(token, ax, ay):
    token = (token or "xy").lower()
    if token in ("x",):
        return [("x", ax)]
    if token in ("y",):
        return [("y", ay)]
    if token in ("xy", "yx", "both", "all"):
        return [("x", ax), ("y", ay)]
    raise ValueError("axis must be x, y, or xy")


def status(ax: Tc100Axis, ay: Tc100Axis):
    x_pos = ax.position_enc_mm()
    y_pos = ay.position_enc_mm()
    x_as = ax.action_status()
    y_as = ay.action_status()
    x_sv = ax.servo_state()
    y_sv = ay.servo_state()
    print(f"X: pos={x_pos:.2f} mm, ActionStatus={x_as}, ServoStatus={x_sv}")
    print(f"Y: pos={y_pos:.2f} mm, ActionStatus={y_as}, ServoStatus={y_sv}")


def scan2d(ax: Tc100Axis, ay: Tc100Axis, speed: int, step: float, maxx: float, maxy: float, dwell: float, serp: bool):
    # Home-less scan: assumes you already homed if you need coordinates
    nx = int(round(maxx / step))
    ny = int(round(maxy / step))
    total = (nx + 1) * (ny + 1)
    k = 0

    t0 = time.time()
    for j in range(ny + 1):
        y = j * step
        ay.move_to(y, speed_percent=speed, wait=True)

        if serp and (j % 2 == 1):
            xs = range(nx, -1, -1)
        else:
            xs = range(0, nx + 1)

        for i in xs:
            x = i * step
            ax.move_to(x, speed_percent=speed, wait=True)

            k += 1
            print(f"[{k}/{total}] X={x:.2f}  Y={y:.2f}  encX={ax.position_enc_mm():.2f}  encY={ay.position_enc_mm():.2f}")

            if dwell > 0:
                time.sleep(dwell)

    print(f"scan2d done. elapsed={time.time()-t0:.1f}s")

def thz_scan(ax: Tc100Axis, ay: Tc100Axis, speed: int, x0: float, y0: float, dwell_s: float):
    """
    太赫茲 5x5 蛇行掃描程序 (更新版：4cm 步距)
    x0, y0: 初始第一張圖片的絕對座標 (mm)
    dwell_s: 每個點停留拍攝的秒數
    """
    rows = 5
    cols = 5
    step_mm = 40.0  # 4 cm = 40 mm
    
    # 方向設定 (依照需求：往左、往上)
    # 註：如果您的馬達實際轉動方向相反，請將這裡的 1 與 -1 互換
    x_dir = -1  # 往左移動：X 座標遞減
    y_dir = -1   # 往上移動：Y 座標遞增
    
    total = rows * cols
    k = 0
    t0 = time.time()
    
    print(f"\n--- 開始太赫茲掃描任務 ---")
    print(f"起點座標: X={x0:.2f}, Y={y0:.2f}")
    print(f"預計拍攝: {total} 張, 每次移動: 4 cm, 每張停留: {dwell_s} 秒\n")

    for r in range(rows):
        # 換列：計算當前列的 Y 絕對座標並移動 (往上)
        curr_y = y0 + r * step_mm * y_dir
        ay.move_to(curr_y, speed_percent=speed, wait=True)
        
        # 決定 X 軸的移動順序
        if r % 2 == 0:
            # 偶數列 (第 1, 3, 5 列)：從右到左 (對應 0, -40, -80, -120, -160)
            c_list = range(0, cols)      
        else:
            # 奇數列 (第 2, 4 列)：從左到右 (對應 -160, -120, -80, -40, 0)
            c_list = range(cols - 1, -1, -1)  
            
        for c in c_list:
            # 計算當前圖片的 X 絕對座標並移動
            curr_x = x0 + c * step_mm * x_dir
            ax.move_to(curr_x, speed_percent=speed, wait=True)
            
            k += 1
            print(f"[拍圖 {k:02d}/{total}] 抵達位置 X={curr_x:.2f}, Y={curr_y:.2f} | 停留 {dwell_s} 秒中...")
            
            # 停下來拍照
            if dwell_s > 0:
                time.sleep(dwell_s)
                
    print(f"\n掃描完成！總耗時: {time.time()-t0:.1f} 秒")
    print("正在將鐵板退回初始位置...")
    
    # 掃描結束，自動回到初始點，方便更換下一塊鐵板
    ax.move_to(x0, speed_percent=speed, wait=True)
    ay.move_to(y0, speed_percent=speed, wait=True)
    print("已回到初始位置，請更換鐵板。\n")

def main():
    bus_x = ModbusASCII(port=X_COM, baudrate=BAUD, parity=PARITY, timeout=1.0, turnaround=0.03)
    bus_y = ModbusASCII(port=Y_COM, baudrate=BAUD, parity=PARITY, timeout=1.0, turnaround=0.03)
    ax = Tc100Axis(bus_x, unit_id=X_ID)
    ay = Tc100Axis(bus_y, unit_id=Y_ID)

    speed = 30  # default speed

    print("=== TC-100 Shell ===")
    print(f"X: {X_COM} id={X_ID} | Y: {Y_COM} id={Y_ID} | baud={BAUD} parity={PARITY}")
    print("Type 'help' for commands.\n")

    try:
        # safe-ish init
        ax.alarm_reset();        ay.alarm_reset()
        ax.servo_on(True);        ay.servo_on(True)
        time.sleep(0.1)
        status(ax, ay)

        while True:
            try:
                line = input("\nTC100> ").strip()
                if not line:
                    continue

                parts = shlex.split(line)
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in ("exit", "quit", "q"):
                    print("Exiting...")
                    break

                if cmd in ("help", "?"):
                    print(HELP)
                    continue

                if cmd == "st":
                    status(ax, ay)
                    continue

                if cmd == "pos":
                    print(f"X={ax.position_enc_mm():.2f} mm, Y={ay.position_enc_mm():.2f} mm")
                    continue

                if cmd == "s":
                    if len(args) != 1:
                        print("Usage: s <0~100>")
                        continue
                    speed = int(float(args[0]))
                    speed = max(0, min(100, speed))
                    print(f"Speed set to {speed}%")
                    continue

                if cmd in ("son", "soff", "rst", "home", "decel", "emg"):
                    axis_token = args[0] if args else "xy"
                    axes = pick_axes(axis_token, ax, ay)

                    if cmd == "son":
                        for _, a in axes: a.servo_on(True)
                        print("Servo ON:", axis_token)
                    elif cmd == "soff":
                        for _, a in axes: a.servo_on(False)
                        print("Servo OFF:", axis_token)
                    elif cmd == "rst":
                        for _, a in axes: a.alarm_reset()
                        print("Alarm reset:", axis_token)
                    elif cmd == "home":
                        for name, a in axes:
                            print(f"Homing {name}...")
                            a.home(wait=True)
                        print("Home done.")
                    elif cmd == "decel":
                        for _, a in axes: a.stop_decel()
                        print("Decel stop:", axis_token)
                    elif cmd == "emg":
                        for _, a in axes: a.stop_emg()
                        print("EMG stop:", axis_token)

                    continue

                if cmd in ("mx", "my"):
                    if len(args) != 1:
                        print("Usage: mx <mm> | my <mm>")
                        continue
                    mm = float(args[0])
                    if cmd == "mx":
                        ax.move_to(mm, speed_percent=speed, wait=True)
                    else:
                        ay.move_to(mm, speed_percent=speed, wait=True)
                    status(ax, ay)
                    continue

                if cmd == "mxy":
                    if len(args) != 2:
                        print("Usage: mxy <x_mm> <y_mm>")
                        continue
                    x_mm = float(args[0]); y_mm = float(args[1])
                    ax.move_to(x_mm, speed_percent=speed, wait=True)
                    ay.move_to(y_mm, speed_percent=speed, wait=True)
                    status(ax, ay)
                    continue

                if cmd in ("dx", "dy"):
                    if len(args) != 1:
                        print("Usage: dx <mm> | dy <mm>")
                        continue
                    d = float(args[0])
                    if cmd == "dx":
                        ax.move_by(d, speed_percent=speed, wait=True)
                    else:
                        ay.move_by(d, speed_percent=speed, wait=True)
                    status(ax, ay)
                    continue

                if cmd == "dxy":
                    if len(args) != 2:
                        print("Usage: dxy <dx_mm> <dy_mm>")
                        continue
                    dx_mm = float(args[0]); dy_mm = float(args[1])
                    ax.move_by(dx_mm, speed_percent=speed, wait=True)
                    ay.move_by(dy_mm, speed_percent=speed, wait=True)
                    status(ax, ay)
                    continue

                if cmd == "scan2d":
                    if len(args) < 3:
                        print("Usage: scan2d <step_mm> <maxx> <maxy> [dwell_s] [serp=0|1]")
                        continue
                    step = float(args[0])
                    maxx = float(args[1])
                    maxy = float(args[2])
                    dwell = float(args[3]) if len(args) >= 4 else 0.0
                    serp = bool(int(args[4])) if len(args) >= 5 else True
                    print(f"scan2d: step={step} maxx={maxx} maxy={maxy} dwell={dwell} serp={serp}")
                    scan2d(ax, ay, speed, step, maxx, maxy, dwell, serp)
                    continue
                
                if cmd == "autoscan":
                    if len(args) < 1:
                        print("Usage: autoscan <dwell_s> [x0_mm] [y0_mm]")
                        print("若未輸入 x0 與 y0，將自動以當前馬達位置作為初始座標。")
                        continue
                    
                    dwell = float(args[0])
                    
                    # 如果使用者有手動輸入初始座標
                    if len(args) >= 3:
                        x0 = float(args[1])
                        y0 = float(args[2])
                    # 如果沒輸入，直接讀取當前編碼器位置當作起點
                    else:
                        x0 = ax.position_enc_mm()
                        y0 = ay.position_enc_mm()
                        
                    thz_scan(ax, ay, speed, x0, y0, dwell)
                    continue

                print("Unknown command. Type 'help'.")

            except KeyboardInterrupt:
                # graceful stop to avoid leaving motors running
                print("\n[CTRL+C] Decel-stopping both axes and returning to prompt...")
                try: ax.stop_decel()
                except Exception: pass
                try: ay.stop_decel()
                except Exception: pass
                time.sleep(0.2)
                continue

            except Exception as e:
                print("Error:", repr(e))

    finally:
        try:
            bus_x.close()
        except Exception:
            pass
        try:
            bus_y.close()
        except Exception:
            pass
        print("Connections closed.")


if __name__ == "__main__":
    main()

# 1.夾著sample蛇行掃描
# TC100> dx dy #初始位置
# TC100> autoscan 3.5 #拍照時間3.5s

# 2.夾著相機掃描
# TC100> dx dy #初始位置
# TC100> stitchscan 2.5 3 20  #停留2.5秒, pixel=3um, 重疊20%