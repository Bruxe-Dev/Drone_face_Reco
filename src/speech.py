import pyttsx3 as pt 

engine = pt.init()

voices = engine.getProperty("voices")

engine.setProperty("voice",voices[1].id)
engine.setProperty("rate", 160)
engine.setProperty("volume",1.0)

def greet(name):
    greeting = f"Hello {name}, nice to meet you!"
    print(f"Speaking: {greeting}")

    engine.say(greeting)
    engine.runAndWait()
