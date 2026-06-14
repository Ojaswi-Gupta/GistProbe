import os
import glob
from gtts import gTTS

def generate_summary_audio(takeaways, session_id):
    """
    Generate an audio summary from the text takeaways using gTTS.
    Returns the path to the saved audio file relative to the static folder.
    """
    if not takeaways:
        return None
        
    # Join the takeaways into a script format for the TTS
    script = "Here are the key takeaways from the article. "
    for i, point in enumerate(takeaways, 1):
        script += f"Point {i}. {point} "
        
    filename = f"audio_{session_id}.mp3"
    filepath = os.path.join("static", filename)
    
    try:
        # Generate speech
        tts = gTTS(text=script, lang='en', slow=False)
        # Ensure static folder exists
        os.makedirs("static", exist_ok=True)
        tts.save(filepath)
        return filename
    except Exception as e:
        print(f"Error generating audio summary: {e}")
        return None

def cleanup_old_audio():
    """Remove old audio files (older than 1 hour) to prevent disk clutter."""
    import time
    now = time.time()
    for f in glob.glob("static/audio_*.mp3"):
        try:
            if os.stat(f).st_mtime < now - 3600:
                os.remove(f)
        except OSError:
            pass
