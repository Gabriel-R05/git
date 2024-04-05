from gtts import gTTS
from pydub import AudioSegment
import webvtt
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

def leer_texto_en_portugues(texto):
    tts = gTTS(text=texto, lang='pt', tld='com.br')
    tts.save("output.mp3")

#####################################################################################################################################
                                                                                                                                    #
    # Cargar el audio generado                                                                                                      #
    audio = AudioSegment.from_mp3("output.mp3")                                                                                     #
                                                                                                                                    #
    # Aumentar la velocidad de reproducción (por ejemplo, 1.2 veces más rápido)                                                     #
    audio_speed_up = audio.speedup(playback_speed=1.38) # Tiempo de velocidad de audio                                              #
                                                                                                                                    #
    # Exportar el audio acelerado                                                                                                   #
    audio_speed_up.export("output_rapido.mp3", format="mp3")                                                                        #
                                                                                                                                    #
#####################################################################################################################################

def procesar_vtt(vtt_path):
    textos = []
    vtt = webvtt.read(vtt_path)
    for caption in vtt:
        if caption.text.strip() != '':
            textos.append(caption.text)
    return textos

def combinar_audio_con_video(audio_path, video_path, output_path):
    # Cargar clips de audio y video
    audio_clip = AudioFileClip(audio_path)
    video_clip = VideoFileClip(video_path)

    # Asegurarse de que el audio tenga la misma duración que el video
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    # Combinar audio y video
    video_con_audio = video_clip.set_audio(audio_clip)

    # Guardar el video resultante
    video_con_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

def text_to_speech_vtt(vtt_path, video_path):
    textos = procesar_vtt(vtt_path)
    texto_completo = ' '.join(textos)
    leer_texto_en_portugues(texto_completo)
    combinar_audio_con_video("output_rapido.mp3", video_path, "video_con_audio.mp4")

# Ejemplo de uso
text_to_speech_vtt('videopt.vtt', 'video_recortado.mp4')

################################################################  PARA COMBINARLO TODO_JUNTO ###################

from moviepy.editor import VideoFileClip, concatenate_videoclips

# Cargar los clips de video
clip_inicio = VideoFileClip("inicio.mp4")
clip_principal = VideoFileClip("video_con_audio.mp4")
clip_final = VideoFileClip("final_recortado.mp4")

# Concatenar los clips en el orden deseado
final_clip = concatenate_videoclips([clip_inicio, clip_principal, clip_final])

# Guardar el video final
final_clip.write_videofile("video_final.mp4")

##############################################################################


from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from torch import  tensor_split
from typing import List, Dict
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.


sf.write("speech1.wav", gTTS["audio"], samplerate=gTTS["sampling_rate"])

