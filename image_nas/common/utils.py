import json
import string
import datetime
import socket
from contextlib import closing
import torch
import secrets


def get_random_string():
    # Random string with the combination of lower and upper case
    char_length = 4
    letters = string.ascii_letters
    result_str = ''.join(secrets.choice(letters) for i in range(char_length))
    now = datetime.datetime.now()
    result_str = now.strftime("%Y%m%d%H%M%S")[2:] + result_str
    return result_str


class PortManager:
    @staticmethod
    def get_free_port():
        for p in range(9000, 9100):
            if not PortManager.is_port_in_use(p):
                return p
        return PortManager.get_random_free_port()

    @staticmethod
    def get_random_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0


class GPUManager:
    @staticmethod
    def is_gpu_available():
        return torch.cuda.is_available()

    @staticmethod
    def get_available_gpu_id():
        if GPUManager.is_gpu_available():
            gpu_count = torch.cuda.device_count()
            for device_num in range(gpu_count):
                memory_usage = GPUManager.cal_gpu_usage(device_num)
                if memory_usage < 1000:
                    return device_num
            return None
        return None

    @staticmethod
    def cal_gpu_usage(gpu_id):
        allocated = round(torch.cuda.memory_allocated(gpu_id) / 1024 ** 2, 1)  # MB
        cached = round(torch.cuda.memory_reserved(gpu_id) / 1024 ** 2, 1)  # MB
        memory_usage = allocated + cached
        return memory_usage


def save_to_json(filename, data: dict):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_from_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data
