import pyaudio





def main(in_device_idx: int|None=None, 
         sample_rate: int=16000,
         chunk_sz: int=512, 
         channels: int=1,
         verbose=False):
    def log(message, tag: str|None=None):
        if verbose:
            if not tag:
                print(f"[LOG] record_donka.main(): {message}")
            if tag:
                print(f"[{tag}] record_donka.main(): {message}")

    p = pyaudio.PyAudio()
    
    # Get devices
    if in_device_idx==None:
        in_device_idx = p.get_default_input_device_info()['index']
    out_device_idx = p.get_default_output_device_info()['index']
    log("Using input device " + p.get_device_info_by_host_api_device_index(0, in_device_idx).get('name'))
    log("Using output device " + p.get_device_info_by_host_api_device_index(0, out_device_idx).get('name'))

    p.terminate()



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="DonKa Recorder",
        description="Retrieve audio inputs from users by following a metronome."
    )

    parser.add_argument("--in-device-idx", type=int, default=-1)

    args = parser.parse_args()
    in_device_idx = args.in_device_idx

    if in_device_idx < 0:
        in_device_idx = None

    main(in_device_idx=in_device_idx, verbose=True)