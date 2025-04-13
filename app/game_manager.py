# app/game_manager.py
import json
import time
import cv2

class NoteObject:
    def __init__(self, note_index, start_time, duration):
        self.note_index = note_index
        self.start_time = start_time
        self.duration = duration
        self.y_position = -100  # Start off-screen
        self.active = False
        self.hit = False

class GameManager:
    """
    Manages the "falling notes" game logic:
      - Loading a song (from a JSON file)
      - Spawning & animating falling notes
      - Checking for key hits
      - Calculating score
    """
    def __init__(self, path_to_song="songs/sample_song.json", speed=80.0):
        """
        Args:
            path_to_song (str): Path to the song JSON data.
            speed (float): How fast notes fall (pixels per second).
        """
        self.speed = speed
        self.notes = self._load_song(path_to_song)
        self.start_time = None
        self.score = 0
        self.combo = 0
        self.missed_notes = 0
        self.judgment_line_y = 180  # Near the bottom of a 200px height view.
        self.hit_window = 0.25  # Seconds window to register a hit

    def _load_song(self, song_path):
        """Load note data from a JSON file."""
        with open(song_path, "r") as f:
            data = json.load(f)
        notes = []
        note_index = 0
        start_time = 0
        duration = 0
        for item in data:
            note_index = item["note"]
            start_time = item["time"]
            duration = item.get("duration", 1.0)
            notes.append(NoteObject(note_index, start_time, duration))
        notes.sort(key=lambda n: n.start_time)
        return notes

    def start_game(self):
        """Initialize game timer and reset score."""
        self.start_time = time.time()
        self.score = 0
        self.combo = 0
        self.missed_notes = 0

    def update(self, dt, white_keys, black_keys, pressed_keys):
        """
        Update game logic once per frame.
        
        Args:
            dt (float): Time elapsed since last frame.
            white_keys (list): Detected white key rectangles.
            black_keys (list): Detected black key rectangles.
            pressed_keys (set): Keys pressed in this frame.
        """
        if self.start_time is None:
            return
        
        current_time = time.time() - self.start_time

        # Activate notes when in time range
        for note in self.notes:
            if not note.active and not note.hit:
                if current_time >= (note.start_time - 1.5):
                    note.active = True

        # Animate falling notes and check for misses
        for note in self.notes:
            if note.active and not note.hit:
                time_until_judgment = note.start_time - current_time
                note.y_position = ( -self.speed * time_until_judgment ) + self.judgment_line_y
                if note.y_position > 220:  # Below the screen.
                    note.active = False
                    note.hit = True
                    self.combo = 0
                    self.missed_notes += 1

        # Detect key hits for active notes near judgment line
        for note in self.notes:
            if note.active and not note.hit:
                time_diff = abs(note.start_time - current_time)
                if time_diff < self.hit_window:
                    if note.note_index in pressed_keys:
                        note.hit = True
                        self.combo += 1
                        points = 100 + (self.combo * 10)
                        self.score += points

    def draw_notes(self, canvas, white_keys, black_keys):
        """
        Draw falling note bars onto the canvas based on their key's position.
        Currently, only white keys are used for simplicity.
        """
        for note in self.notes:
            if not note.active or note.hit:
                continue
            rect = None
            if note.note_index < 7 and note.note_index < len(white_keys):
                rect = white_keys[note.note_index]
            # For simplicity, we'll only render white-key notes in this example.
            if rect is not None:
                x, y, w, h = rect
                note_x = x
                note_w = w
                note_y = int(note.y_position)
                note_h = 20  # Fixed height; can be adjusted by duration
                color = (0, 0, 255)  # Red for the falling note
                cv2.rectangle(canvas, (note_x, note_y),
                              (note_x + note_w, note_y + note_h),
                              color, -1)

    def get_score(self):
        return self.score
