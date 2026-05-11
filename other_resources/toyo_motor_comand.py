# tc100_shell.py
# Interactive resident CLI controller for 2x TC-100 (X/Y)
#
# Requires: tc100.py (ModbusASCII, Tc100Axis)
# pip install pyserial

import time
import shlex
from tc100 import ModbusASCII, Tc100Axis

# ======== your setup ========
X_COM, X_ID = "COM4", 1
Y_COM, Y_ID = "COM5", 2
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
                                 Example: scan2d 2 300 300 0.05 1

Notes:
- All moves are blocking (wait=True) by default in this shell.
- Ctrl+C will try to decel-stop both axes and return to prompt.
""".strip()


def pick_axes(token: str | None, ax, ay):
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
        ax.alarm_reset(); ay.alarm_reset()
        ax.servo_on(True); ay.servo_on(True)
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
