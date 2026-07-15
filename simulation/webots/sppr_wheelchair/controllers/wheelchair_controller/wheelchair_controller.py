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


# =========================================================
# 2. Lấy thiết bị
# =========================================================

left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
inertial_unit = robot.getDevice("inertial unit")

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


# =========================================================
# 5. Tham số điều khiển
# =========================================================

FORWARD_SPEED = 3.0
BACKWARD_SPEED = 2.5

TURN_SPEED_FAST = 1.5
TURN_SPEED_SLOW = 0.45

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


def set_motor_speeds(
    left_speed: float,
    right_speed: float
) -> None:
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)


def stop_robot() -> None:
    global current_state, target_yaw

    set_motor_speeds(0.0, 0.0)

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
        set_motor_speeds(0.0, 0.0)

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
finally:
    try:
        set_motor_speeds(0.0, 0.0)
    finally:
        udp_socket.close()

    print("UDP receiver closed and robot stopped")
