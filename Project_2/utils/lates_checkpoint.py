import glob

def find_latest_checkpoint():
    files = glob.glob("saved_params/*.msgpack")
    if not files:
        return None, 0
    
    files.sort(key=lambda f: int(f.split('/')[-1].split('_')[0]))
    
    latest = files[-1]
    number = int(latest.split('/')[-1].split('_')[0])

    return latest, number