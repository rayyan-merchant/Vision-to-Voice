 
import easyocr, pyttsx3 
reader = easyocr.Reader(["en"]) 
print("EasyOCR ready") 
engine = pyttsx3.init() 
engine.say("Vision to Voice system online.") 
engine.runAndWait() 
print("TTS ready") 