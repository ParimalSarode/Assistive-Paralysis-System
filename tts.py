import subprocess
import sys
import queue
import threading

speech_queue = queue.Queue()

def speech_worker():
    script = (
        "import pyttsx3, sys; "
        "engine = pyttsx3.init(); "
        "engine.setProperty('rate', 150); "
        "engine.say(sys.argv[1]); "
        "engine.runAndWait()"
    )
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            subprocess.run([sys.executable, "-c", script, text], creationflags=0x08000000)
        except Exception as e:
            print(f"TTS Error: {e}")
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak_text(text: str):
    speech_queue.put(text)
