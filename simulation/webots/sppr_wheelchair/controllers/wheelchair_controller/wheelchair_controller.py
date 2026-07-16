import json
import math
import socket

from controller import Robot


# =========================================================
# 1. Khởi tạo Webots
# =========================================================

robot = Robot()
timestep = int(robot.getBasicTimeStep())

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 5006
TELEMETRY_INTERVAL_SECONDS = 0.1
VALID_UDP_COMMANDS = {
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
}

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    udp_socket.bind((UDP_HOST, UDP_PORT))
except OSError as error:
    udp_socket.close()
    raise RuntimeError(
        f"Failed to bind UDP receiver on {UDP_HOST}:{UDP_PORT}: {error}"
    ) from error

udp_socket.setblocking(False)
print(f"UDP receiver initialized on {UDP_HOST}:{UDP_PORT}")

telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
telemetry_socket.setblocking(False)
print(f"Telemetry sender initialized for {TELEMETRY_HOST}:{TELEMETRY_PORT}")


# =========================================================
# 2. Lấy thiết bị
# =========================================================

left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
inertial_unit = robot.getDevice("inertial unit")
gps = robot.getDevice("gps")

keyboard = robot.getKeyboard()
keyboard.enable(timestep)


# =========================================================
# 3. Kiểm tra thiết bị
# =========================================================

if left_motor is None:
    raise RuntimeError('Motor "left wheel" was not found.')

if right_motor is None:
    raise RuntimeError('Motor "right wheel" was not found.')

if inertial_unit is None:
    raise RuntimeError(
        'InertialUnit "inertial unit" was not found.'
    )


# =========================================================
# 4. Cấu hình thiết bị
# =========================================================

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

inertial_unit.enable(timestep)
if gps is not None:
    gps.enable(timestep)
else:
    print('Warning: GPS "gps" was not found; telemetry is disabled.')


# =========================================================
# 5. Tham số điều khiển
# =========================================================

FORWARD_SPEED = 3.0
BACKWARD_SPEED = 2.5

TURN_SPEED_FAST = 1.5
TURN_SPEED_SLOW = 0.45
WHEEL_ACCELERATION_LIMIT = 3.0

TURN_ANGLE_DEGREES = 90.0
ANGLE_TOLERANCE = math.radians(2.0)
SLOW_DOWN_THRESHOLD = math.radians(15.0)


# =========================================================
# 6. Trạng thái
# =========================================================

STATE_STOP = "STOP"
STATE_FORWARD = "FORWARD"
STATE_BACKWARD = "BACKWARD"
STATE_TURN_LEFT = "TURN_LEFT"
STATE_TURN_RIGHT = "TURN_RIGHT"

current_state = STATE_STOP
target_yaw = None
current_left_speed = 0.0
current_right_speed = 0.0
target_left_speed = 0.0
target_right_speed = 0.0
last_telemetry_time = -TELEMETRY_INTERVAL_SECONDS
telemetry_warning_printed = False


# =========================================================
# 7. Hàm tiện ích
# =========================================================

def normalize_angle(angle: float) -> float:
    """
    Chuẩn hóa góc về khoảng [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def get_current_yaw() -> float:
    """
    Trả về yaw hiện tại của robot, đơn vị radian.
    """
    return inertial_unit.getRollPitchYaw()[2]


def send_telemetry() -> None:
    global last_telemetry_time, telemetry_warning_printed

    if gps is None:
        return

    now = robot.getTime()
    if now - last_telemetry_time < TELEMETRY_INTERVAL_SECONDS:
        return

    position = gps.getValues()
    payload = {
        "timestamp": now,
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "yaw": get_current_yaw(),
        "motion_state": current_state,
        "left_velocity": current_left_speed,
        "right_velocity": current_right_speed,
    }
    try:
        telemetry_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            (TELEMETRY_HOST, TELEMETRY_PORT),
        )
    except (BlockingIOError, OSError) as error:
        if not telemetry_warning_printed:
            print(f"Warning: failed to send telemetry: {error}")
            telemetry_warning_printed = True
    else:
        last_telemetry_time = now
        telemetry_warning_printed = False


def set_motor_speeds(
    left_speed: float,
    right_speed: float,
    immediate: bool = False,
) -> None:
    global current_left_speed, current_right_speed
    global target_left_speed, target_right_speed

    target_left_speed = left_speed
    target_right_speed = right_speed

    if immediate:
        current_left_speed = left_speed
        current_right_speed = right_speed
        left_motor.setVelocity(current_left_speed)
        right_motor.setVelocity(current_right_speed)


def move_toward(
    current_value: float,
    target_value: float,
    max_change: float,
) -> float:
    difference = target_value - current_value
    if abs(difference) <= max_change:
        return target_value
    return current_value + math.copysign(max_change, difference)


def update_motor_speeds() -> None:
    global current_left_speed, current_right_speed

    delta_seconds = timestep / 1000.0
    max_change = WHEEL_ACCELERATION_LIMIT * delta_seconds
    current_left_speed = move_toward(
        current_left_speed,
        target_left_speed,
        max_change,
    )
    current_right_speed = move_toward(
        current_right_speed,
        target_right_speed,
        max_change,
    )
    left_motor.setVelocity(current_left_speed)
    right_motor.setVelocity(current_right_speed)


def stop_robot() -> None:
    global current_state, target_yaw

    set_motor_speeds(0.0, 0.0, immediate=True)

    current_state = STATE_STOP
    target_yaw = None

    print("Action: STOP")


def move_forward() -> None:
    global current_state, target_yaw

    target_yaw = None
    current_state = STATE_FORWARD

    set_motor_speeds(
        FORWARD_SPEED,
        FORWARD_SPEED
    )

    print("Action: MOVE_FORWARD")


def move_backward() -> None:
    global current_state, target_yaw

    target_yaw = None
    current_state = STATE_BACKWARD

    set_motor_speeds(
        -BACKWARD_SPEED,
        -BACKWARD_SPEED
    )

    print("Action: MOVE_BACKWARD")


def start_turn_left(
    degrees: float = TURN_ANGLE_DEGREES
) -> None:
    global current_state, target_yaw

    current_yaw = get_current_yaw()

    target_yaw = normalize_angle(
        current_yaw + math.radians(degrees)
    )

    current_state = STATE_TURN_LEFT

    print(f"Action: TURN_LEFT {degrees:.0f} degrees")


def start_turn_right(
    degrees: float = TURN_ANGLE_DEGREES
) -> None:
    global current_state, target_yaw

    current_yaw = get_current_yaw()

    target_yaw = normalize_angle(
        current_yaw - math.radians(degrees)
    )

    current_state = STATE_TURN_RIGHT

    print(f"Action: TURN_RIGHT {degrees:.0f} degrees")


def update_turn() -> None:
    """
    Cập nhật quá trình quay theo yaw.
    Khi đạt góc đích, robot tự dừng.
    """
    global current_state, target_yaw

    if current_state not in {
        STATE_TURN_LEFT,
        STATE_TURN_RIGHT,
    }:
        return

    if target_yaw is None:
        stop_robot()
        return

    current_yaw = get_current_yaw()

    angle_error = normalize_angle(
        target_yaw - current_yaw
    )

    absolute_error = abs(angle_error)

    if absolute_error <= ANGLE_TOLERANCE:
        set_motor_speeds(0.0, 0.0, immediate=True)

        print(
            "Target angle reached | "
            f"remaining error: "
            f"{math.degrees(angle_error):.2f} degrees"
        )

        current_state = STATE_STOP
        target_yaw = None
        return

    if absolute_error <= SLOW_DOWN_THRESHOLD:
        turn_speed = TURN_SPEED_SLOW
    else:
        turn_speed = TURN_SPEED_FAST

    if current_state == STATE_TURN_LEFT:
        set_motor_speeds(
            -turn_speed,
            turn_speed
        )

    elif current_state == STATE_TURN_RIGHT:
        set_motor_speeds(
            turn_speed,
            -turn_speed
        )


# =========================================================
# 8. Xử lý keyboard
# =========================================================

def handle_key(key: int) -> None:
    """
    Xử lý một phím từ Webots Keyboard.
    """
    if key == -1:
        return

    # Loại bỏ modifier bits nếu có.
    normalized_key = key & 0xFF

    if normalized_key in {
        ord("W"),
        ord("w"),
    }:
        move_forward()

    elif normalized_key in {
        ord("S"),
        ord("s"),
    }:
        move_backward()

    elif normalized_key in {
        ord("A"),
        ord("a"),
    }:
        if current_state not in {
            STATE_TURN_LEFT,
            STATE_TURN_RIGHT,
        }:
            start_turn_left(90.0)

    elif normalized_key in {
        ord("D"),
        ord("d"),
    }:
        if current_state not in {
            STATE_TURN_LEFT,
            STATE_TURN_RIGHT,
        }:
            start_turn_right(90.0)

    elif normalized_key == ord(" "):
        stop_robot()


# =========================================================
# 9. Xử lý UDP command
# =========================================================

def handle_udp_command(command: str) -> None:
    if command == "MOVE_FORWARD":
        move_forward()
    elif command == "MOVE_BACKWARD":
        move_backward()
    elif command == "TURN_LEFT":
        start_turn_left(90.0)
    elif command == "TURN_RIGHT":
        start_turn_right(90.0)
    elif command == "STOP":
        stop_robot()


# =========================================================
# 10. Main loop
# =========================================================

print("Keyboard control started")
print("W     : move forward")
print("S     : move backward")
print("A     : turn left 90 degrees")
print("D     : turn right 90 degrees")
print("Space : stop")
print(f"Wheel acceleration limit: {WHEEL_ACCELERATION_LIMIT:.1f} rad/s^2")

try:
    while robot.step(timestep) != -1:
        try:
            packet, sender_address = udp_socket.recvfrom(1024)
        except BlockingIOError:
            pass
        else:
            try:
                udp_command = packet.decode("utf-8").strip().upper()
            except UnicodeDecodeError as error:
                print(
                    "Warning: ignored UDP packet with invalid UTF-8 "
                    f"from {sender_address}: {error}"
                )
            else:
                if udp_command in VALID_UDP_COMMANDS:
                    print(
                        f"UDP command received from {sender_address}: "
                        f"{udp_command}"
                    )
                    handle_udp_command(udp_command)
                else:
                    print(
                        f"Warning: ignored invalid UDP command "
                        f"from {sender_address}: {udp_command!r}"
                    )

        key = keyboard.getKey()

        while key != -1:
            handle_key(key)
            key = keyboard.getKey()

        update_turn()
        update_motor_speeds()
        send_telemetry()
finally:
    try:
        set_motor_speeds(0.0, 0.0, immediate=True)
    finally:
        try:
            udp_socket.close()
        finally:
            telemetry_socket.close()

    print("UDP sockets closed and robot stopped")
