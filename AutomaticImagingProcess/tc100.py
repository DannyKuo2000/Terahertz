# tc100.py
# TC-100 Modbus ASCII control via pyserial
# pip install pyserial

from __future__ import annotations
import time
from dataclasses import dataclass
import serial


# ---------- Modbus ASCII core ----------
def _lrc(payload: bytes) -> int:
    s = sum(payload) & 0xFF
    return (-s) & 0xFF


def _to_hex(bs: bytes) -> bytes:
    return bs.hex().upper().encode()


def _from_hex(bhex: bytes) -> bytes:
    return bytes.fromhex(bhex.decode())


class ModbusASCII:
    def __init__(
        self,
        port: str = "COM4",
        baudrate: int = 19200,
        parity: str = "N",
        timeout: float = 1.0,
        turnaround: float = 0.02,
        bytesize: int = 8,
        stopbits: int = 1,
    ):
        par = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN, "O": serial.PARITY_ODD}[parity]
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=bytesize,
            parity=par,
            stopbits=stopbits,
            timeout=timeout,
        )
        self.turnaround = turnaround

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _xfer(self, unit: int, pdu: bytes) -> bytes:
        payload = bytes([unit]) + pdu
        frame = b":" + _to_hex(payload + bytes([_lrc(payload)])) + b"\r\n"

        self.ser.reset_input_buffer()
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(self.turnaround)

        deadline = time.time() + (self.ser.timeout or 0)
        buf = b""
        while time.time() < deadline:
            ch = self.ser.read(1)
            if not ch:
                continue
            buf += ch

            # keep only from first ':'
            idx = buf.find(b":")
            if idx == -1:
                continue
            if idx > 0:
                buf = buf[idx:]

            if buf.startswith(b":") and buf.endswith(b"\r\n"):
                break

        if not (buf.startswith(b":") and buf.endswith(b"\r\n")):
            raise OSError("No response")

        raw = _from_hex(buf[1:-2])  # strip ':' and CRLF
        if len(raw) < 3:
            raise OSError(f"Too short: {raw!r}")
        if _lrc(raw[:-1]) != raw[-1]:
            raise OSError("LRC mismatch")

        # exception response
        if raw[1] & 0x80:
            code = raw[2] if len(raw) > 2 else -1
            raise OSError(f"Exception from slave, code={code}")

        return raw

    # FC03
    def read_holding_registers(self, unit: int, address: int, count: int) -> list[int]:
        pdu = bytes([0x03, (address >> 8) & 0xFF, address & 0xFF, (count >> 8) & 0xFF, count & 0xFF])
        raw = self._xfer(unit, pdu)

        bc = raw[2]
        data = raw[3 : 3 + bc]
        if bc != 2 * count or len(data) != 2 * count:
            raise OSError(f"Byte count mismatch: bc={bc}, count={count}")

        return [(data[i * 2] << 8) | data[i * 2 + 1] for i in range(count)]

    # FC06
    def write_register(self, unit: int, address: int, value: int):
        value &= 0xFFFF
        pdu = bytes([0x06, (address >> 8) & 0xFF, address & 0xFF, (value >> 8) & 0xFF, value & 0xFF])
        _ = self._xfer(unit, pdu)

    # FC16
    def write_registers(self, unit: int, address: int, values: list[int]):
        vals = [v & 0xFFFF for v in values]
        count = len(vals)
        body = b"".join(bytes([(v >> 8) & 0xFF, v & 0xFF]) for v in vals)
        pdu = bytes(
            [0x10, (address >> 8) & 0xFF, address & 0xFF, (count >> 8) & 0xFF, count & 0xFF, 2 * count]
        ) + body
        _ = self._xfer(unit, pdu)


# ---------- Register Map ----------
@dataclass(frozen=True)
class Tc100Map:
    ADDR_ACTION_STATUS: int = 0x1000
    ADDR_SERVO_STATUS: int = 0x100C
    ADDR_ECD_POS: int = 0x100A

    ADDR_INC_AMOUNT: int = 0x2000  # 32-bit
    ADDR_ABS_AMOUNT: int = 0x2002  # 32-bit

    ADDR_SERVO: int = 0x2011
    ADDR_MOV_SPEED: int = 0x2014
    ADDR_MOV_TYPE: int = 0x201E  # 0=INC,1=ABS,3=HOME,6=RST,8=DECEL,9=EMG

    SCALE_POS: float = 100.0  # 0.01 mm units (based on your tests)


class Tc100Axis:
    """
    High-level axis wrapper.
    NOTE: Your controller uses 32-bit registers as Hi-word first: [hi, lo].
    """

    def __init__(self, bus: ModbusASCII, unit_id: int = 1, m: Tc100Map = Tc100Map()):
        self.bus = bus
        self.uid = unit_id
        self.m = m

    # --- low level ---
    def _r(self, addr: int, n: int = 1) -> list[int]:
        return self.bus.read_holding_registers(self.uid, addr, n)

    def _w(self, addr: int, val: int):
        self.bus.write_register(self.uid, addr, val & 0xFFFF)

    def _r32(self, addr: int) -> int:
        hi, lo = self._r(addr, 2)  # ✅ Hi-word first
        val = (hi << 16) | lo
        if val & 0x80000000:
            val -= 0x100000000
        return val

    def _w32(self, addr: int, val: int):
        if val < 0:
            val = (val + (1 << 32)) & 0xFFFFFFFF
        hi, lo = (val >> 16) & 0xFFFF, val & 0xFFFF
        self.bus.write_registers(self.uid, addr, [hi, lo])  # ✅ Hi-word first

    # --- status helpers ---
    def action_status(self) -> int:
        return self._r(self.m.ADDR_ACTION_STATUS, 1)[0]

    def is_in_motion(self) -> bool:
        # from your measurement:
        # stopped -> 0, moving -> 1
        return self.action_status() == 1

    def servo_state(self) -> int:
        return self._r(self.m.ADDR_SERVO_STATUS, 1)[0]

    def position_enc_mm(self) -> float:
        return self._r32(self.m.ADDR_ECD_POS) / self.m.SCALE_POS

    # --- commands ---
    def alarm_reset(self):
        self._w(self.m.ADDR_MOV_TYPE, 6)
        time.sleep(0.05)

    def servo_on(self, on: bool = True):
        # Your controller shows ServoStatus=1 when ON, but this control register semantics
        # can vary by firmware. Keep original behavior; adjust if needed.
        self._w(self.m.ADDR_SERVO, 0 if on else 1)

    def home(self, wait: bool = True, timeout: float = 180.0):
        self._w(self.m.ADDR_MOV_TYPE, 3)
        if wait:
            self.wait_until_stop(timeout=timeout)

    def move_by(self, delta_mm: float, speed_percent: int | None = None, wait: bool = False, timeout: float = 180.0):
        if speed_percent is not None:
            self._w(self.m.ADDR_MOV_SPEED, int(speed_percent))

        raw = int(round(delta_mm * self.m.SCALE_POS))
        self._w32(self.m.ADDR_INC_AMOUNT, raw)
        self._w(self.m.ADDR_MOV_TYPE, 0)

        if wait:
            self.wait_until_stop(timeout=timeout)

    def move_to(self, pos_mm: float, speed_percent: int | None = None, wait: bool = False, timeout: float = 180.0):
        if speed_percent is not None:
            self._w(self.m.ADDR_MOV_SPEED, int(speed_percent))

        raw = int(round(pos_mm * self.m.SCALE_POS))
        self._w32(self.m.ADDR_ABS_AMOUNT, raw)
        self._w(self.m.ADDR_MOV_TYPE, 1)

        if wait:
            self.wait_until_stop(timeout=timeout)

    def wait_until_stop(self, poll: float = 0.02, timeout: float = 180.0, start_timeout: float = 1.0) -> bool:
        """
        ActionStatus-based wait:
        1) wait until it starts moving (ActionStatus=1), or start_timeout expires
        2) wait until it stops (ActionStatus=0)
        """
        t0 = time.time()

        # wait for start (optional)
        while True:
            if self.is_in_motion():
                break
            if time.time() - t0 > start_timeout:
                break
            time.sleep(poll)

        # wait for stop
        while True:
            if not self.is_in_motion():
                return True
            if time.time() - t0 > timeout:
                raise TimeoutError("Motion timeout")
            time.sleep(poll)

    def stop_decel(self):
        self._w(self.m.ADDR_MOV_TYPE, 8)

    def stop_emg(self):
        self._w(self.m.ADDR_MOV_TYPE, 9)
