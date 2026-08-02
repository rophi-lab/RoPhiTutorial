from dynamixel_sdk import *
import time
import threading
from utils.mode import HandControlMode
import numpy as np

# echo 1 | sudo tee -a /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# also need to set LATENCY_TIMER to 1ms in port_handler.py file of dynamixel_sdk


def deg2rad(deg):
    return deg * 3.141592653589793 / 180.0


def rad2deg(rad):
    return rad * 180.0 / 3.141592653589793


def write_byte(port_handler, packet_handler, id, address, value, debug=False):
    result, error = packet_handler.write1ByteTxRx(port_handler, id, address, value)
    if result != COMM_SUCCESS:
        print(
            f"Write Error: {packet_handler.getTxRxResult(result)} at {id} at {address}"
        )
    elif error != 0:
        print(
            f"Hardware Error: {packet_handler.getRxPacketError(error)} at {id} at {address}"
        )
    else:
        if debug:
            print(f"Successfully wrote {value} to {id} at {address}")


def data_to_bytes(data, sizes):
    data_bytes = [0] * sum(sizes)
    i = 0
    k = 0
    for size in sizes:
        for j in range(size):
            data_bytes[i] = (data[k] >> (j * 8)) & 0xFF
            i += 1
        k += 1
    return data_bytes


def write_data(port_handler, packet_handler, id, address, data, sizes, debug=False):
    data_bytes = data_to_bytes(data, sizes)
    result, error = packet_handler.writeTxRx(
        port_handler, id, address, sum(sizes), data_bytes
    )
    if result != COMM_SUCCESS:
        print(
            f"Write Error: {packet_handler.getTxRxResult(result)} at {id} at {address}"
        )
    elif error != 0:
        print(
            f"Hardware Error: {packet_handler.getRxPacketError(error)} at {id} at {address}"
        )
    else:
        if debug:
            print(f"Successfully wrote {data} to {id} at {address}")


def read_bytes(port_handler, packet_handler, id, address, length, debug=False):
    data_read, dxl_comm_result, dxl_error = packet_handler.readTxRx(
        port_handler, id, address, length
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(
            f"Read Error: {packet_handler.getTxRxResult(dxl_comm_result)} at {id} at {address}"
        )
    elif dxl_error != 0:
        print(
            f"Hardware Error: {packet_handler.getRxPacketError(dxl_error)} at {id} at {address}"
        )
    else:
        if debug:
            print(f"Data Read: {data_read}")
        return data_read


def read_data(port_handler, packet_handler, id, address, length, sizes, debug=False):
    data_read = read_bytes(port_handler, packet_handler, id, address, length)
    if data_read is not None:
        data = []
        i = 0
        for size in sizes:
            data.append(
                int.from_bytes(
                    bytes(data_read[i : i + size]), byteorder="little", signed=True
                )
            )
            i += size
        return data


# INDIRECT ADDRESSING
""" 
individual motor indirect addressing
need to configure so it can read just two bytes of position and velocity instead of 4 bytes
motor addr: torque en (64, 1), des curr (102, 2), des pos (116, 2), curr (126, 2), vel (128, 2), pos (132, 2)
indir addr:           168                170                174            178         
indir data:           224                225                229            231        

sync tables
writing should be set up to write des curr and des pos for each motor 
read address 

Each address stores 1 byte, adress is EEPROM so set it on init 

sync table:
read: 33 bytes: 4* (2 curr + 2 vel + 2 pos) + 9*1 pressure
write: 16 bytes: 4* (2 curr + 2 pos) 
total: 49 bytes per sync table

indirect mapping: 49*5 = 245 bytes ez fit 
have to manually set every indirect address

"""


class Robotis5FHardware:
    # communication
    # DEVICENAME = '/dev/ttyUSB0'
    BAUDRATE = 4000000
    PROTOCOL_VERSION = 2.0

    TORQUE_CONSTANT = 1.2  # Nm/A #1.2 from robotis

    MOTOR_PER_SYNC = 4
    NUM_SYNCS = 5

    # Hand sync addresses
    ADDR_HAND_SYNC_EN = 70
    ADDR_HAND_SYNC_READ_START = 1078  # , 1276, 1474, 1672, 1870]
    ADDR_HAND_SYNC_WRITE_DATA = 1150  # , 1348, 1546, 1744, 1942]

    ADDR_HAND_SYNC_ID = 1024
    HAND_SYNC_ID_SIZE = 1
    ADDR_HAND_SYNC_READ_ADDRESS = 1030
    HAND_SYNC_READ_ADDRESS_SIZE = 2
    ADDR_HAND_SYNC_READ_SIZE = 1042
    HAND_SYNC_READ_SIZE_SIZE = 2
    ADDR_HAND_SYNC_WRITE_ADDRESS = 1054
    HAND_SYNC_WRITE_ADDRESS_SIZE = 2
    ADDR_HAND_SYNC_WRITE_SIZE = 1066
    HAND_SYNC_WRITE_SIZE_SIZE = 2

    HAND_SYNC_WRITE_WIDTH = 198  # width from synctable 1 to synctable 2

    # Hand indirect addressing
    ADDR_HAND_INDIRECT_ADDR_START = 122  # start from here 2 byte addresses
    ADDR_HAND_INDIRECT_ADDR_READ_START = ADDR_HAND_INDIRECT_ADDR_START + 5 * (2 * 4 * 4)
    """
    writing 
    each motor as 4 bytes of command data (des_cur, des_pos) , each synctable has 4 motors so 16 addresses per synctable, each address is 2 bytes 
    122 -> 1150 
    124 -> 1151 
    ...
    122 + 2*4*4 = 154 -> 1150 + 16 - 1 = 1165

    reading
    each motor read 6 bytes + 9 for pressure sensor data, each synctable has 4 motors so 33 addresses per synctable, each address is 2 bytes
    start at READ_START then add 33 for each synctable 
    """
    ADDR_HAND_INDIRECT_DATA_START = 634  # start from here 1 byte data
    ADDR_HAND_INDIRECT_DATA_READ_START = ADDR_HAND_INDIRECT_DATA_START + 4 * 20

    # contiguous chunk for writing and contiguous chunk for reading

    # Motor addresses
    ADDR_MOTOR_TORQUE_EN = 64
    ADDR_MOTOR_CONTROL_MODE = 11
    ADDR_MOTOR_GOAL_CURRENT = 102
    ADDR_MOTOR_GOAL_POSITION = 116
    ADDR_MOTOR_INDIRECT_ADDR = 168
    ADDR_MOTOR_INDIRECT_DATA_READ_CURR = 229
    ADDR_MOTOR_INDIRECT_DATA_WRITE_CURR = 225
    ADDR_MOTOR_READ_CURRENT = 126
    ADDR_MOTOR_READ_VELOCITY = 128
    ADDR_MOTOR_READ_POSITION = 132
    ADDR_MOTOR_KP = 84
    ADDR_MOTOR_KD = 80

    CONTROL_MODE_CURRENT = 0
    CONTROL_MODE_CURRENT_BASED_POSITION = 5

    def __init__(self, devicename, zero_pos, control_mode: HandControlMode):
        self.port_handler = PortHandler(devicename)
        self.packet_handler = PacketHandler(Robotis5FHardware.PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            print("[Hardware] Failed to open the port")
            quit()

        if not self.port_handler.setBaudRate(Robotis5FHardware.BAUDRATE):
            print("[Hardware] Failed to change the baudrate")
            quit()

        self.control_mode = control_mode

        # IDs
        self.HAND_ID = 110
        self.MOTOR_IDS = [
            111,
            112,
            113,
            114,
            116,
            117,
            118,
            119,
            121,
            122,
            123,
            124,
            126,
            127,
            128,
            129,
            131,
            132,
            133,
            134,
        ]

        # positions
        self.MIN_POS = [
            88.59,
            87.63,
            88.51,
            88.42,
            90,
            177,
            88,
            87,
            129,
            177,
            87,
            87,
            158,
            177,
            87,
            87,
            166,
            177,
            87,
            87,
        ]

        self.MAX_POS = [
            226.67,
            216.21,
            272.37,
            272.99,
            194,
            311,
            271,
            271,
            223,
            308,
            271,
            271,
            259,
            310,
            271,
            271,
            295,
            314,
            271,
            271,
        ]

        self.zero_pos = zero_pos  # in degrees

        self._lock = threading.Lock()

        self.NUM_JOINTS = 20

        # motoro configuration
        self.flip_motors = [0]

        # sensor data
        self._currents = [0] * self.NUM_JOINTS
        self._velocities = [0] * self.NUM_JOINTS
        self._positions = [0] * self.NUM_JOINTS
        self._pressures = [[0] * 9] * 5
        self._raw_currents_buf = [0] * self.NUM_JOINTS
        self._raw_velocities_buf = [0] * self.NUM_JOINTS
        self._raw_positions_buf = [0.0] * self.NUM_JOINTS
        self._raw_pressures_buf = [[0] * 9 for _ in range(Robotis5FHardware.NUM_SYNCS)]

        # commanded data
        # Raw command sent to hardware (int mA in current mode, raw position units in position mode).
        self._command = None
        # Desired impedance terms for CURRENT_CONTROL; evaluated continuously in IO thread.
        self._q_des = None
        self._qd_des = None
        self._tau_des = None
        self._kp_des = None
        self._kd_des = None
        self._new_command_available = False

        # threading
        self._stop_event = threading.Event()
        self._io_thread = None

    def start(self):
        # initialize hand
        self.disable_sync()
        self.motor_memory_mapping()
        print("Disabling torque to initialize...")
        self.disable_torque()
        if self.control_mode == HandControlMode.POSITION_CONTROL:
            self.set_position_control()
            self.set_kp_kd(kp=300, kd=600, max_current=120)  # only for position control
        elif self.control_mode == HandControlMode.CURRENT_CONTROL:
            self.set_current_control()

        self.hand_memory_mapping()

        print("Enabling torque and sync...")
        self.enable_torque()
        self.enable_sync()

        if self._io_thread is None:
            self._stop_event.clear()
            self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
            self._io_thread.start()

    def hand_memory_mapping(self):
        # setup sync table, should not do this every time but w/e for now
        # IDs already set
        # read addresses
        print("Setting up sync table read address...")
        self.set_sync_table_values(
            Robotis5FHardware.ADDR_HAND_SYNC_READ_ADDRESS,
            Robotis5FHardware.HAND_SYNC_READ_ADDRESS_SIZE,
            [Robotis5FHardware.ADDR_MOTOR_INDIRECT_DATA_READ_CURR] * 4 + [72],
        )

        # read size
        print("Setting up sync table read size...")
        self.set_sync_table_values(
            Robotis5FHardware.ADDR_HAND_SYNC_READ_SIZE,
            Robotis5FHardware.HAND_SYNC_READ_SIZE_SIZE,
            [6] * 4 + [9],
        )

        # write address
        print("Setting up sync table write address...")
        self.set_sync_table_values(
            Robotis5FHardware.ADDR_HAND_SYNC_WRITE_ADDRESS,
            Robotis5FHardware.HAND_SYNC_WRITE_ADDRESS_SIZE,
            [Robotis5FHardware.ADDR_MOTOR_INDIRECT_DATA_WRITE_CURR] * 4 + [0],
        )

        # write size
        print("Setting up sync table write size...")
        self.set_sync_table_values(
            Robotis5FHardware.ADDR_HAND_SYNC_WRITE_SIZE,
            Robotis5FHardware.HAND_SYNC_WRITE_SIZE_SIZE,
            [4, 4, 4, 4, 0, 0],
        )

        # configure hand indirect addressing
        data = []
        for i in range(5):
            data += [
                Robotis5FHardware.ADDR_HAND_SYNC_WRITE_DATA
                + i * Robotis5FHardware.HAND_SYNC_WRITE_WIDTH
                + j
                for j in range(16)
            ]  # map to the 16 bytes starting at 1150 which are the indirect data for the sync tables
        sizes = [2] * len(data)

        write_data(
            self.port_handler,
            self.packet_handler,
            self.HAND_ID,
            Robotis5FHardware.ADDR_HAND_INDIRECT_ADDR_START,
            data,
            sizes,
        )

        # configure hand indirect addressing for reading
        data = []
        for i in range(5):
            data += [
                Robotis5FHardware.ADDR_HAND_SYNC_READ_START
                + i * Robotis5FHardware.HAND_SYNC_WRITE_WIDTH
                + j
                for j in range(33)
            ]  # map to the 33 bytes starting at 1078 which are the read data for the sync tables
        sizes = [2] * len(data)

        write_data(
            self.port_handler,
            self.packet_handler,
            self.HAND_ID,
            Robotis5FHardware.ADDR_HAND_INDIRECT_ADDR_READ_START,
            data,
            sizes,
        )

    def motor_memory_mapping(self):
        data = [
            Robotis5FHardware.ADDR_MOTOR_TORQUE_EN,
            Robotis5FHardware.ADDR_MOTOR_GOAL_CURRENT,
            Robotis5FHardware.ADDR_MOTOR_GOAL_CURRENT + 1,
            Robotis5FHardware.ADDR_MOTOR_GOAL_POSITION,
            Robotis5FHardware.ADDR_MOTOR_GOAL_POSITION + 1,
            Robotis5FHardware.ADDR_MOTOR_READ_CURRENT,
            Robotis5FHardware.ADDR_MOTOR_READ_CURRENT + 1,
            Robotis5FHardware.ADDR_MOTOR_READ_VELOCITY,
            Robotis5FHardware.ADDR_MOTOR_READ_VELOCITY + 1,
            Robotis5FHardware.ADDR_MOTOR_READ_POSITION,
            Robotis5FHardware.ADDR_MOTOR_READ_POSITION + 1,
        ]
        sizes = [2] * len(data)

        for id in self.MOTOR_IDS:
            write_data(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_INDIRECT_ADDR,
                data,
                sizes,
            )

    def set_sync_table_values(self, addr: int, size: int, data: list):
        # sets all the sync tables for all 5 fingers, for a given initial sync table address and field size
        sizes = [size] * len(data)
        for i in range(Robotis5FHardware.NUM_SYNCS):
            write_data(
                self.port_handler,
                self.packet_handler,
                self.HAND_ID,
                addr + i * Robotis5FHardware.HAND_SYNC_WRITE_WIDTH,
                data,
                sizes,
            )

    def set_motors_indirect_addr(self, data, sizes):
        for id in self.MOTOR_IDS:
            write_data(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_INDIRECT_ADDR,
                data,
                sizes,
            )

    def stop(self):
        if self._io_thread is not None:
            self._stop_event.set()
            self._io_thread.join()
            self._io_thread = None

        self.port_handler.is_using = False  # this is so sus
        self.disable_sync()
        time.sleep(0.1)
        self.disable_torque()

    def _io_loop(self):
        last_print_time = time.time()
        loop_count = 0
        read_sizes = ([2] * 12 + [1] * 9) * Robotis5FHardware.NUM_SYNCS
        deg_per_unit = 0.087891
        rad_per_unit = deg_per_unit * 3.141592653589793 / 180.0
        revPmin_per_unit = 0.22888
        # Per-second profiler for understanding loop bottlenecks across machines.
        prof = {
            "loop_total": 0.0,
            "read_call": 0.0,
            "unpack_and_convert": 0.0,
            "state_copy_lock": 0.0,
            "cmd_fetch_lock": 0.0,
            "write_call": 0.0,
            "loop_max": 0.0,
            "read_max": 0.0,
            "write_max": 0.0,
            "samples": 0,
        }

        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()
            # read
            try:
                raw_currents = self._raw_currents_buf
                raw_velocities = self._raw_velocities_buf
                raw_positions = self._raw_positions_buf
                raw_pressures = self._raw_pressures_buf

                # data is sets of 12 2 byte values (3 per motor) then 9 1 byte values then repeat  so 21 data entires per sync
                t_read_start = time.perf_counter()
                data = read_data(
                    self.port_handler,
                    self.packet_handler,
                    self.HAND_ID,
                    Robotis5FHardware.ADDR_HAND_INDIRECT_DATA_READ_START,
                    33 * 5,
                    read_sizes,
                )
                read_dt = time.perf_counter() - t_read_start
                prof["read_call"] += read_dt
                if read_dt > prof["read_max"]:
                    prof["read_max"] = read_dt

                if data is None:
                    continue
                t_unpack_start = time.perf_counter()

                for i in range(Robotis5FHardware.NUM_SYNCS):
                    for j in range(Robotis5FHardware.MOTOR_PER_SYNC):
                        current = data[i * 21 + j * 3]
                        velocity = data[i * 21 + j * 3 + 1]
                        position = data[i * 21 + j * 3 + 2]

                        motor_index = i * Robotis5FHardware.MOTOR_PER_SYNC + j

                        raw_currents[motor_index] = (
                            current * 0.001 * Robotis5FHardware.TORQUE_CONSTANT
                        )  # convert to Nm
                        raw_velocities[motor_index] = (
                            (velocity * revPmin_per_unit) * 2 * 3.141592653589793 / 60.0
                        )
                        raw_positions[motor_index] = (
                            position * rad_per_unit
                        ) - deg2rad(self.zero_pos[motor_index])

                        if motor_index in self.flip_motors:
                            raw_velocities[motor_index] = -raw_velocities[motor_index]
                            raw_positions[motor_index] = -raw_positions[motor_index]
                            raw_currents[motor_index] = -raw_currents[motor_index]

                    raw_pressures[i][:] = data[i * 21 + 12 : i * 21 + 21]
                prof["unpack_and_convert"] += time.perf_counter() - t_unpack_start

                t_lock_copy_start = time.perf_counter()
                with self._lock:
                    self._currents[:] = raw_currents
                    self._velocities[:] = raw_velocities
                    self._positions[:] = raw_positions
                    for i in range(Robotis5FHardware.NUM_SYNCS):
                        self._pressures[i][:] = raw_pressures[i]
                prof["state_copy_lock"] += time.perf_counter() - t_lock_copy_start

            except Exception as e:
                print(f"[RobotisHardware] Exception in reading thread: {e}")

            # write:
            # Keep sending the latest command even when no new command was posted.
            # For CURRENT_CONTROL, recompute current from latest measured q/qd every loop.
            cmd = None
            t_cmd_lock_start = time.perf_counter()
            with self._lock:
                if (
                    self.control_mode == HandControlMode.CURRENT_CONTROL
                    and self._q_des is not None
                ):
                    curr_pos = np.array(self._positions)
                    curr_vel = np.array(self._velocities)
                    current = (
                        1000
                        * (
                            self._tau_des
                            + self._kp_des * (self._q_des - curr_pos)
                            + self._kd_des * (self._qd_des - curr_vel)
                        )
                        / Robotis5FHardware.TORQUE_CONSTANT
                    )
                    max_current = 10000  # mA
                    current = np.clip(current, -max_current, max_current).astype(int)
                    if len(self.flip_motors) > 0:
                        current[self.flip_motors] = -current[self.flip_motors]
                    self._command = current.copy()
                if self._command is not None:
                    cmd = self._command.copy()
                if self._new_command_available:
                    self._new_command_available = False
            prof["cmd_fetch_lock"] += time.perf_counter() - t_cmd_lock_start

            if cmd is not None:
                try:
                    t_write_start = time.perf_counter()
                    self.write_motor_command(cmd)
                    write_dt = time.perf_counter() - t_write_start
                    prof["write_call"] += write_dt
                    if write_dt > prof["write_max"]:
                        prof["write_max"] = write_dt
                    # pass
                except Exception as e:
                    print(f"[RobotisHardware] Exception in writing thread: {e}")

            loop_dt = time.perf_counter() - t_loop_start
            prof["loop_total"] += loop_dt
            prof["samples"] += 1
            if loop_dt > prof["loop_max"]:
                prof["loop_max"] = loop_dt

            loop_count += 1
            loop_print_dt = 0.05
            if time.time() - last_print_time >= loop_print_dt:
                samples = max(prof["samples"], 1)
                # print(
                #     "[RobotisHardware] IO Loop "
                #     f"{loop_count/loop_print_dt:.1f} Hz | "
                #     f"loop avg/max: {prof['loop_total']/samples*1e3:.3f}/{prof['loop_max']*1e3:.3f} ms | "
                #     f"read avg/max: {prof['read_call']/samples*1e3:.3f}/{prof['read_max']*1e3:.3f} ms | "
                #     f"unpack avg: {prof['unpack_and_convert']/samples*1e3:.3f} ms | "
                #     f"state_lock avg: {prof['state_copy_lock']/samples*1e3:.3f} ms | "
                #     f"cmd_lock avg: {prof['cmd_fetch_lock']/samples*1e3:.3f} ms | "
                #     f"write avg/max: {prof['write_call']/samples*1e3:.3f}/{prof['write_max']*1e3:.3f} ms"
                # )
                loop_count = 0
                last_print_time = time.time()
                for k in prof:
                    prof[k] = 0.0 if k != "samples" else 0

            # time.sleep(0.002)

    def enable_sync(self):
        write_byte(
            self.port_handler,
            self.packet_handler,
            self.HAND_ID,
            Robotis5FHardware.ADDR_HAND_SYNC_EN,
            1,
        )

    def disable_sync(self):
        write_byte(
            self.port_handler,
            self.packet_handler,
            self.HAND_ID,
            Robotis5FHardware.ADDR_HAND_SYNC_EN,
            0,
        )

    def enable_torque(self):
        # self.disable_sync()
        for id in self.MOTOR_IDS:
            write_byte(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_TORQUE_EN,
                1,
            )
        # self.enable_sync()

    # TODO should look into using broadcasting
    def disable_torque(self):
        # self.disable_sync()
        for id in self.MOTOR_IDS:
            write_byte(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_TORQUE_EN,
                0,
            )
        # self.enable_sync()

    def set_current_control(self):
        # self.disable_sync()
        for id in self.MOTOR_IDS:
            write_byte(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_CONTROL_MODE,
                Robotis5FHardware.CONTROL_MODE_CURRENT,
            )
        # self.enable_sync()

    def set_position_control(self):
        # self.disable_sync()
        for id in self.MOTOR_IDS:
            write_byte(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_CONTROL_MODE,
                Robotis5FHardware.CONTROL_MODE_CURRENT_BASED_POSITION,
            )
        # self.enable_sync()

    def set_kp_kd(self, kp, kd, max_current):
        for id in self.MOTOR_IDS:
            write_data(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_KP,
                [kp],
                [2],
            )
            write_data(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_KD,
                [kd],
                [2],
            )
            write_data(
                self.port_handler,
                self.packet_handler,
                id,
                Robotis5FHardware.ADDR_MOTOR_GOAL_CURRENT,
                [max_current],
                [2],
            )

    def write_motor_command(self, command):
        raw_command = command.copy()
        data = [1] * len(raw_command) * 2
        data[::2] = raw_command
        data[1::2] = raw_command
        sizes = [2] * len(raw_command) * 2
        write_data(
            self.port_handler,
            self.packet_handler,
            self.HAND_ID,
            Robotis5FHardware.ADDR_HAND_INDIRECT_DATA_START,
            data,
            sizes,
        )

    def command(self, q, q_dot, tau, kp, kd):
        # based on contorl mode selected write appropriate data
        # where should the reading motor position happen? Probably in a thread in this class
        if self.control_mode == HandControlMode.POSITION_CONTROL:
            position_des = rad2deg(q.copy())
            position_des += self.zero_pos  # add zero pos offset
            deg_per_unit = 0.087891
            raw_command = (position_des / deg_per_unit).astype(int)
            with self._lock:
                self._command = raw_command
                self._new_command_available = True
        elif self.control_mode == HandControlMode.CURRENT_CONTROL:
            # Save desired impedance terms; IO thread evaluates control continuously.
            with self._lock:
                self._q_des = np.array(q).copy()
                self._qd_des = np.array(q_dot).copy()
                self._tau_des = np.array(tau).copy()
                self._kp_des = np.array(kp).copy()
                self._kd_des = np.array(kd).copy()
                self._new_command_available = True

    def get_motor_data(self):
        with self._lock:
            return (
                np.array(self._positions.copy()),
                np.array(self._velocities.copy()),
                np.array(self._currents.copy()),
            )
