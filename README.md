# Keyzz

Piano Sheet Detector and Interactive Player 
--------------------------------------------------------
This module provides a modularized implementation of the piano detection and key-press audio
playback system. The code is organized into distinct classes:
 
  • AudioManager        - Initializes the audio subsystem, loads sounds, handles debouncing and playback.
  • PianoDetector       - Detects the piano sheet corners, piano keys, and dot positions.
  • PerspectiveTransformer
                        - Applies perspective transformation and provides point transformation utilities.
  • Stabilizer          - Provides stabilization logic so that the detected piano sheet remains consistent.
  • Visualizer          - Handles all drawing and overlay operations.
  • PianoApp            - The main application orchestrating capture, detection, stabilization, and interaction.
 
Usage:
    Run the script directly. The "sounds" folder must be available in the same directory as this file,
    containing the corresponding .wav files.