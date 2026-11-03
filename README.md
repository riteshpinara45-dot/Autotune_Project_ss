# Autotune & Audio Filtering System (Signals & Systems Project)

A real-time and file-based Autotune and audio filtering system implemented in Python, applying digital signal processing (DSP) concepts including filtering, autocorrelation-based pitch detection, and overlap-add (OLA) pitch shifting with phase matching.

## 🌟 Key Features

- **DSP Signal Chain**:
  - **High-pass Filter**: Removes DC offset and low-frequency rumble below 80 Hz.
  - **Low-pass Filter**: Removes high-frequency noise above 8 kHz (voice intelligibility band).
  - **Pre-emphasis Filter**: Shelving boost above 3 kHz to assist pitch detection.
- **Autotune Engine**:
  - **Pitch Detection**: YIN autocorrelation algorithm (standalone NumPy implementation).
  - **Pitch Correction**: Snaps detected pitch to user-selected musical scales (Chromatic, Major, Minor, Pentatonic, etc.) with configurable correction strength.
  - **Pitch Shifting**: Overlap-Add (OLA) with cross-correlation phase alignment to prevent phase jumps at block boundaries.
- **Interfaces**:
  - **Web Application (`app.py`)**: Modern Flask web UI (`web_static/index.html`) supporting browser recording, file uploads, waveform visualization, and parameter tuning.
  - **Desktop GUI (`autotune_gui.py`)**: Tkinter & Matplotlib desktop interface with live audio monitoring and spectrogram/waveform plotting.

---

## 📁 Repository Structure

```
├── app.py                            # Flask web application server
├── autotune_gui.py                   # Desktop Tkinter GUI application
├── autotune_system_main_phase.py     # Core DSP engine (filters, YIN, OLA pitch shifter)
├── web_static/
│   └── index.html                    # Web UI frontend
├── scratch/                          # Verification and testing scripts
├── requirements.txt                  # Python dependencies
└── Differentiation_integration_amp.pdf # Coursework / Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.9+ installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Applications

#### Option A: Web Interface
Start the Flask web server:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

#### Option B: Desktop GUI
Launch the desktop application:
```bash
python autotune_gui.py
```

---

## 📚 Technologies Used

- **Python 3**
- **NumPy & SciPy**: Digital signal processing (IIR/FIR filters, autocorrelation, OLA)
- **SoundFile & SoundDevice**: Audio I/O and streaming
- **Flask**: Web backend API
- **Tkinter & Matplotlib**: Desktop interface and visualizations
