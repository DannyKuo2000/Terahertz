import time
import pyautogui
import tkinter as tk
from threading import Thread, Event
from pynput import keyboard

# Set up
POINTS = [(800, 300), (800, 450)]
PERIOD_READY = 3
PERIOD1 = 1
PERIOD2 = 1

# Control events
running = Event()   # 是否正在執行 click loop
running.clear()     # 一開始先暫停

stopped = Event()   # 是否結束整個程式
stopped.clear()


# Red frame for aiming
def draw_targets():
    root = tk.Tk()
    root.attributes("-topmost", True)  # 讓視窗在最上面
    root.overrideredirect(True)  # 移除視窗邊框
    root.attributes('-transparentcolor', 'white')  # 設定透明顏色

    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()

    canvas = tk.Canvas(
        root,
        width=screen_width,
        height=screen_height,
        bg='white',
        highlightthickness=0
    )
    canvas.pack(fill='both', expand=True)

    for x, y in POINTS:
        size = 6
        canvas.create_rectangle(x-size//2, y-size//2, x+size//2, y+size//2, outline='red', width=2)

    root.mainloop()


# Keyboard control
def on_press(key):
    try:
        if key == keyboard.KeyCode.from_char('k'):
            if running.is_set():
                running.clear()
                print("Click loop: STOPPED")
            else:
                running.set()
                print("Click loop: RUNNING")

        elif key == keyboard.Key.esc:
            stopped.set()
            running.clear()
            print("Program: EXIT")
            return False  # 停止 keyboard listener

    except Exception as e:
        print(f"Keyboard error: {e}")


def click_loop():
    print("Press K to start/stop clicking.")
    print("Press ESC to exit.")

    while not stopped.is_set():

        # 等待 K 開始
        if not running.is_set():
            time.sleep(PERIOD_READY)
            continue

        # 等待 PERIOD1
        start_wait = time.perf_counter()
        while time.perf_counter() - start_wait < PERIOD1:
            if not running.is_set() or stopped.is_set():
                break
            time.sleep(0.01)

        if not running.is_set() or stopped.is_set():
            continue

        # 點擊第一個位置
        pyautogui.click(POINTS[0][0], POINTS[0][1])

        # 等待 PERIOD2
        start_wait = time.perf_counter()
        while time.perf_counter() - start_wait < PERIOD2:
            if not running.is_set() or stopped.is_set():
                break
            time.sleep(0.01)

        if not running.is_set() or stopped.is_set():
            continue

        # 點擊第二個位置
        pyautogui.click(POINTS[1][0], POINTS[1][1])


if __name__ == "__main__":

    # Start the red frame in a separate thread
    frame_thread = Thread(target=draw_targets, daemon=True)
    frame_thread.start()

    # Start keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    # Start the clicking loop
    click_loop()

    # 等待 keyboard listener 結束
    keyboard_listener.join()