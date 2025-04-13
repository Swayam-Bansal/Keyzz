# app/game_manager.py
import json
import time
import cv2
import numpy as np
import logging

class NoteObject:
    def __init__(self, note_index, start_time, duration):
        self.note_index = note_index
        self.start_time = start_time
        self.duration = duration
        self.y_position = -100  # Start off-screen
        self.active = False
        self.hit = False
        self.missed = False  # New field to track if note was missed
        self.prev_pressed_keys = set()

class GameManager:
    """
    Manages the "falling notes" game logic:
      - Loading a song (from a JSON file)
      - Spawning & animating falling notes
      - Checking for key hits
      - Calculating score and lives
    """
    def __init__(self, path_to_song="songs/twinkle_twinkle_little_star.json", speed=80.0, total_lives=3):
        """
        Args:
            path_to_song (str): Path to the song JSON data.
            speed (float): How fast notes fall (pixels per second).
            total_lives (int): Number of lives the player starts with.
        """
        self.speed = speed
        self.notes = self._load_song(path_to_song)
        self.start_time = None
        self.score = 0
        self.combo = 0
        self.lives = total_lives
        self.total_lives = total_lives
        self.judgment_line_y = 180  # Near the bottom of a 200px height view.
        self.perfect_hit_window = 0.1  # Seconds window for "perfect" hit
        self.good_hit_window = 0.2  # Seconds window for "good" hit
        self.hit_window = 0.3  # Maximum seconds window to register any hit
        self.game_over = False
        self.last_hit_judgment = ""
        self.last_hit_time = 0
        self.judgment_display_time = 1.0  # How long to show judgment text
        self.recently_hit_notes = set()  # Track recently hit notes to prevent double hits

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    def _load_song(self, song_path):
        """Load note data from a JSON file."""
        try:
            with open(song_path, "r") as f:
                data = json.load(f)
            notes = []
            for item in data:
                note_index = item["note"]
                start_time = item["time"]
                duration = item.get("duration", 1.0)
                notes.append(NoteObject(note_index, start_time, duration))
            notes.sort(key=lambda n: n.start_time)
            return notes
        except Exception as e:
            logging.error(f"Error loading song file: {e}")
            return []

    def start_game(self):
        """Initialize game timer and reset score."""
        self.start_time = time.time()
        self.score = 0
        self.combo = 0
        self.lives = self.total_lives
        self.game_over = False
        self.last_hit_judgment = ""
        self.last_hit_time = 0
        # Reset all notes
        for note in self.notes:
            note.active = False
            note.hit = False
            note.missed = False
        
        self.recently_hit_notes = set()
        logging.info("Game started/restarted")
        

    def update(self, dt, white_keys, black_keys, pressed_keys):
    # --- Early exit if the game is already over ---
    # (If you still want animation even after game over, you might move this check later)
        if self.game_over:
            return

        # --- Activate notes based on current time ---
        current_time = time.time() - self.start_time
        for note in self.notes:
            if not note.active and not note.hit and not note.missed:
                if current_time >= (note.start_time - 1.5):
                    note.active = True

        # --- Process key hits using new key logic ---
        if not hasattr(self, 'prev_pressed_keys'):
            self.prev_pressed_keys = set()

        # Identify only keys that are newly pressed this frame
        new_keys = pressed_keys - self.prev_pressed_keys
        self.prev_pressed_keys = pressed_keys.copy()

        # Build a dictionary of the closest active note for each key (only for notes that have been activated)
        closest_notes = {}
        for i, note in enumerate(self.notes):
            if note.active and not note.hit and not note.missed:
                time_diff = note.start_time - current_time
                if -self.hit_window <= time_diff <= 1.5:
                    if (note.note_index not in closest_notes or 
                        abs(time_diff) < abs(closest_notes[note.note_index][1])):
                        closest_notes[note.note_index] = (i, time_diff)
        
        # Process hits for new key presses only
        for key in new_keys:
            if key in closest_notes:
                note_idx, time_diff = closest_notes[key]
                note = self.notes[note_idx]
                abs_time_diff = abs(time_diff)
                
                if abs_time_diff <= self.perfect_hit_window:
                    points = 100
                    self.combo += 1
                    self.last_hit_judgment = "PERFECT!"
                elif abs_time_diff <= self.good_hit_window:
                    points = 70
                    self.combo += 1
                    self.last_hit_judgment = "GOOD!"
                elif abs_time_diff <= self.hit_window:
                    points = 50
                    self.combo += 1
                    self.last_hit_judgment = "OK"
                else:
                    if time_diff > 0:
                        points = max(10, int(50 * (1 - abs_time_diff / 1.5)))
                        self.combo += 1
                        self.last_hit_judgment = "EARLY"
                    else:
                        points = 10
                        self.combo = 0
                        self.last_hit_judgment = "LATE"
                
                combo_bonus = min(self.combo * 5, 50)
                self.score += points + combo_bonus
                
                note.hit = True
                self.last_hit_time = current_time
                logging.info(f"Hit note {note.note_index} with judgment: {self.last_hit_judgment}")

        # --- Check for missed notes and update lives ---
        for note in self.notes:
            if note.active and not note.hit and not note.missed:
                time_diff = note.start_time - current_time
                if time_diff < -self.hit_window:
                    note.active = False
                    note.missed = True
                    self.combo = 0
                    self.lives = max(self.lives - 1, 0)
                    self.last_hit_judgment = "MISS"
                    self.last_hit_time = current_time
                    logging.info(f"Note {note.note_index} missed. Lives remaining: {self.lives}")
                    if self.lives == 0 and not self.game_over:
                        self.game_over = True
                        logging.info("Game over: No lives remaining")

        # --- Animate falling notes ---
        for note in self.notes:
            if note.active and not note.hit and not note.missed:
                time_until_judgment = note.start_time - current_time
                note.y_position = (-self.speed * time_until_judgment) + self.judgment_line_y


    def draw_notes(self, canvas, white_keys, black_keys):
        """
        Draw falling note bars onto the canvas based on their key's position.
        Also draws the judgment line and judgment text.
        """
        if canvas is None or white_keys is None:
            return canvas
            
        # Draw judgment line
        cv2.line(canvas, (0, self.judgment_line_y), (self.canvas_width, self.judgment_line_y), 
                 (255, 0, 0), 2)  # Blue line

        # Draw active notes
        for note in self.notes:
            if not note.active or note.hit or note.missed:
                continue
                
            rect = None
            if note.note_index < 7 and note.note_index < len(white_keys):
                rect = white_keys[note.note_index]
            elif note.note_index >= 7:
                # Map the note index to a visual index for black keys
                black_key_visual_idx = None
                for visual_idx, note_idx in self.BLACK_KEY_VISUAL_MAP().items():
                    if note_idx == note.note_index and visual_idx < len(black_keys):
                        black_key_visual_idx = visual_idx
                        break
                
                if black_key_visual_idx is not None:
                    rect = black_keys[black_key_visual_idx]
                    
            if rect is not None:
                x, y, w, h = rect
                note_x = x
                note_w = w
                note_y = int(note.y_position)
                note_h = 20  # Fixed height; can be adjusted by duration
                
                # Gradient color based on distance to judgment line
                distance_to_line = abs(note_y + note_h - self.judgment_line_y)
                if distance_to_line < 20:
                    # Close to line - bright green
                    color = (0, 255, 0)
                elif distance_to_line < 50:
                    # Medium distance - yellow
                    color = (0, 255, 255)
                else:
                    # Far from line - cyan
                    color = (255, 255, 0)
                    
                cv2.rectangle(canvas, (note_x, note_y),
                              (note_x + note_w, note_y + note_h),
                              color, -1)
                
        # Draw judgment text if it's still within display time
        current_time = time.time() - self.start_time if self.start_time else 0
        if current_time - self.last_hit_time < self.judgment_display_time:
            text_size = cv2.getTextSize(self.last_hit_judgment, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (self.canvas_width - text_size[0]) // 2
            text_y = self.judgment_line_y - 30
            
            # Choose color based on judgment
            if self.last_hit_judgment == "PERFECT!":
                color = (0, 255, 0)  # Green
            elif self.last_hit_judgment == "GOOD!":
                color = (0, 255, 255)  # Yellow
            elif self.last_hit_judgment == "OK":
                color = (255, 255, 0)  # Cyan
            elif self.last_hit_judgment == "MISS":
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White
                
            cv2.putText(canvas, self.last_hit_judgment, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        
        return canvas

    def draw_score_and_lives(self, canvas):
        """Draw score, combo, and lives on the canvas."""
        if canvas is None:
            return canvas
            
        # Draw score
        cv2.putText(canvas, f"Score: {self.score}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw combo
        if self.combo > 1:
            cv2.putText(canvas, f"Combo: {self.combo}x", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw lives
        lives_text = f"Lives: {self.lives}/{self.total_lives}"
        text_size = cv2.getTextSize(lives_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(canvas, lives_text, (self.canvas_width - text_size[0] - 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw game over text if applicable
        if self.game_over:
            self._draw_game_over(canvas)
            
        return canvas
        
    def _draw_game_over(self, canvas):
        """Draw game over screen."""
        game_over_text = "GAME OVER"
        text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (self.canvas_width - text_size[0]) // 2
        text_y = self.canvas_height // 2
        
        # Draw semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.canvas_width, self.canvas_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        # Draw game over text
        cv2.putText(canvas, game_over_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Draw final score
        final_score_text = f"Final Score: {self.score}"
        score_text_size = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        score_x = (self.canvas_width - score_text_size[0]) // 2
        score_y = text_y + 50
        cv2.putText(canvas, final_score_text, (score_x, score_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw restart instructions
        restart_text = "Press 'r' to restart"
        restart_text_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        restart_x = (self.canvas_width - restart_text_size[0]) // 2
        restart_y = score_y + 40
        cv2.putText(canvas, restart_text, (restart_x, restart_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def get_score(self):
        return self.score
        
    def get_lives(self):
        return self.lives
        
    def BLACK_KEY_VISUAL_MAP(self):
        """Returns a mapping from visual index to note index for black keys."""
        return {0: 7, 1: 8, 2: 9, 3: 10, 4: 11}
        
    @property
    def canvas_width(self):
        return 700  # Match the canvas width from PianoApp
        
    @property
    def canvas_height(self):
        return 200  # Match the canvas height from PianoApp