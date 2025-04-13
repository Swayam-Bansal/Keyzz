# 🎹 Keyzz: Virtual Piano Experience

**Keyzz** transforms your laptop into an interactive, gamified piano, allowing you to enjoy playing music without needing an actual piano or additional hardware. Leveraging computer vision, machine learning, and real-time audio feedback, Keyzz provides an immersive piano-playing experience right from your home.

---

## 🚀 What is Keyzz?

Keyzz is an innovative, vision-based virtual piano that utilizes OpenCV and MediaPipe to detect printed piano keys on paper, interpret finger movements, and produce real-time musical notes. The integrated gamification elements, inspired by popular rhythm games, make learning and playing piano exciting and accessible for everyone, everywhere.

### 🌟 Key Features:

- **Vision-Based Key Detection**: Detects and maps printed piano keys using just your laptop's webcam.
- **Real-Time Audio Playback**: Plays notes instantly based on finger placement and movement.
- **Interactive Game Mode**: Engage with falling notes similar to rhythm-based games, enhancing your musical learning experience.
- **Instant Calibration**: Quickly stabilizes and calibrates detection for consistent and reliable performance.

---

## ⚙️ Getting Started

Follow these simple steps to set up and use Keyzz on your computer.

### 📋 Prerequisites

- Python >= 3.8
- Webcam-equipped Laptop
- Printed Piano Keys Template ([Download here](#))

### 🔧 Installation

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/yourusername/keyzz.git
   cd keyzz
   ```

2. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Download Piano Sounds**:

   - Ensure piano note sound files (e.g., `C4.wav`, `D4.wav`) are placed in the `sounds/` directory.

### 🎼 Using Keyzz

1. **Print the Piano Sheet**:

   - Use the provided template and print it on standard A4 sheets.

2. **Run the Application**:

   ```sh
   python PianoApp.py
   ```

3. **Calibration**:

   - Position your printed piano sheet within the camera frame clearly.
   - Hold still momentarily for automatic calibration and stabilization.

4. **Start Playing!**:

   - Once calibrated, start playing notes by placing your fingers over the printed keys.
   - Engage in the interactive game mode to follow along and practice rhythmically.

### 🕹 Controls:

- Press `q` to quit the application.
- Press `r` to recalibrate and restart the game.

---

## 🎮 Gamified Experience

Keyzz integrates an intuitive rhythm game that challenges you to play the correct notes as they fall down the screen, encouraging practice and musical improvement. Score points, maintain streaks, and improve your piano skills without any external piano hardware!

---

## 💡 Why Keyzz?

Keyzz breaks down barriers to musical education and enjoyment by eliminating the need for costly instruments and specialized equipment. Ideal for learners, hobbyists, and music enthusiasts, Keyzz brings the joy of piano-playing to everyone, everywhere.

Enjoy your journey with Keyzz—your portable, virtual, and interactive piano companion!

---

## 📬 Feedback & Contributions

We appreciate your feedback and welcome contributions. Feel free to open issues, submit pull requests, or contact the maintainers for improvements and discussions.

Happy Playing! 🎶🎹


[![Watch the video](https://img.youtube.com/vi/BO39RYbYyUE/maxresdefault.jpg)(https://www.youtube.com/watch?v=BO39RYbYyUE)]
