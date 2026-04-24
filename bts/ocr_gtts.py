from gtts import gTTS
import os

text = "Hello Rijaa! EasyOCR and TTS are ready!"
tts = gTTS(text=text, lang='en')  
tts.save("output.mp3")           
os.system("start output.mp3")