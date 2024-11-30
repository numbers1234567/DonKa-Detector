import keyboard
from audio_func.utility import retrieve_audio_inputs,DON,KA,LEFT,RIGHT,DonkaCode
from audio_func.audio_input_detector import AudioInputDetector
from typing import Tuple
import configparser
from dataclasses import dataclass

import time

import queue
import threading

@dataclass
class DonKaKeymap:
    lka:  str
    ldon: str
    rdon: str
    rka:  str

def presskey(keymap: DonKaKeymap, donka_code: DonkaCode):
    if ( RIGHT|DON ) == donka_code: keyboard.send(keyboard.key_to_scan_codes(keymap.rdon)[0])
    if (  LEFT|DON ) == donka_code: keyboard.send(keyboard.key_to_scan_codes(keymap.ldon)[0])
    if ( RIGHT|KA  ) == donka_code: keyboard.send(keyboard.key_to_scan_codes(keymap.rka)[0])
    if (  LEFT|KA  ) == donka_code: keyboard.send(keyboard.key_to_scan_codes(keymap.lka)[0])

# Assumes that the press_queue is increasing
def presskey_queue(keymap: DonKaKeymap, press_queue: queue.Queue[Tuple[float, DonkaCode]]):
    while True:
        timing,donka_code = press_queue.get()
        if donka_code < 0:
            break

        # Wait until it is time
        t = time.time()
        time.sleep(max(0, timing - t + 0.1))

        print(donka_code, time.time())
        presskey(keymap, donka_code)


def main(keymap: DonKaKeymap, user_audio_dir: str, sr: int=16000):
    train_x, train_y, noise_stat = retrieve_audio_inputs(target_dir=user_audio_dir, sr=sr)
    
    # Start the keypress worker
    press_queue: queue.Queue[Tuple[float,DonkaCode]] = queue.Queue()
    keypress_worker = threading.Thread(target=presskey_queue, 
                                       args=(keymap, press_queue))
    
    keypress_worker.start()

    detector = AudioInputDetector(train_x, train_y, noise_stat, lambda donka_code,timing: press_queue.put((timing,donka_code)))

    thread = detector.start()

    input()

    press_queue.put((time.time(), -1))
    detector.stop()
    thread.join()
    keypress_worker.join()

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    key_config = config["keymap"]
    path_config = config["paths"]
    audio_config = config["audio"]

    main(
        DonKaKeymap(
            lka =key_config["LeftKa"],
            ldon=key_config["LeftDon"],
            rdon=key_config["RightDon"],
            rka =key_config["RightKa"],
        ),
        user_audio_dir=path_config["AudioInputs"],
        sr=int(audio_config["SampleRate"]),
    )