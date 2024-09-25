import time

def text_streamer(text, delay=0.05):
    for word in text.split(" "):
        yield word
        yield " "
        time.sleep(delay)