# app/audio_manager.py
import pygame
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class AudioManager:
    """Handles audio initialization, sound loading, and debounced playback."""
    def __init__(self, sounds_folder: str = "sounds", debounce_interval: float = 0.2):
        pygame.mixer.init()
        self.sounds_folder = sounds_folder
        self.debounce_interval = debounce_interval
        self.notes = {
            0: 'C4', 1: 'D4', 2: 'E4', 3: 'F4', 4: 'G4', 5: 'A4', 6: 'B4',
            7: 'C#4', 8: 'D#4', 9: 'F#4', 10: 'G#4', 11: 'A#4'
        }
        self.sounds = {}
        self.last_press_time = {}
        self.load_sounds()

    def load_sounds(self):
        """Load sound files from the specified folder."""
        logging.info("Loading sounds...")
        for key, note in self.notes.items():
            sound_path = f"{self.sounds_folder}/{note}.wav"
            try:
                self.sounds[key] = pygame.mixer.Sound(sound_path)
                logging.info(f"Loaded sound for {note}")
            except pygame.error:
                logging.warning(f"Sound file for {note} not found at {sound_path}.")
        logging.info("Sound loading complete.")

    def play_sound(self, key: int):
        """Play the sound corresponding to the given key if debounced."""
        current_time = time.time()
        if key in self.sounds and (current_time - self.last_press_time.get(key, 0)) > self.debounce_interval:
            self.sounds[key].play()
            self.last_press_time[key] = current_time
            logging.info(f"Played: {self.notes.get(key, 'Unknown')}")
