import argparse
import io
import speech_recognition as sr
import torch
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
from TranscriptionWindow import TranscriptionWindow
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="Device to use for CTranslate2 inference",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_float16", "float16", "int16", "float32"])
    parser.add_argument("--input_lang", default='en', help="Language of the input audio.", type=str)
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model.")
    parser.add_argument("--translate_lang", default="en", help="Language to translate the transcription into.", type=str)
    parser.add_argument("--translate", action='store_true',
                        help="Translate the transcription to the specified language.")
    parser.add_argument("--threads", default=0,
                        help="Number of threads used for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real-time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
                             
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
    
    if args.model == "large":
        args.model = "large-v2"    
    
    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model + ".en"
        
    device = args.device
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = args.compute_type
    cpu_threads = args.threads
    
    audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    window = TranscriptionWindow()
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name 
    transcription = ['']
    last_displayed_text = ""
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                text = ""
                translate = args.translate
                
                # Use Whisper to transcribe and translate (if specified)
                segments, info = audio_model.transcribe(temp_file, language=args.input_lang, task="translate" if translate else "transcribe", target_lang=args.translate_lang if translate else None)
                
                for segment in segments:
                    text += segment.text

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                result = ''.join(transcription[-10:])
                
                # Update the window only if the text has changed
                if result != last_displayed_text:
                    window.update_text([result], args.translate_lang if translate else args.input_lang)
                    last_displayed_text = result
                
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
