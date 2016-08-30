from gtts import gTTS
import os
tts = gTTS(text='Bonjour, tout le monde!', lang='fr')
tts.save("good.mp3")
os.system("mpg321 good.mp3")
