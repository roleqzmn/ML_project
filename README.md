# IML Project
This project uses machine learning to decide whether a person is allowed to enter based on their voice. It converts audio recordings into spectrograms and trains CNN models to classify speakers as authorized or unauthorized.

## Milestone 1

### Audio conversion
`wav_convereter.ps1` converts audio files to WAV, including files in all subfolders. Original files stay unchanged.

- **Installation:**

  Requires Windows 10/11. FFmpeg is downloaded automatically on first use.
- **Usage:**
  Eun directly in PowerShell:
  ```powershell
  .\wav_converter.ps1 -InputFolder "C:\recordings"
  ```
- **Output:**
  Converted files are saved in a `WAV` folder inside the selected folder, preserving the subfolder structure. Existing WAV files are skipped.

### Audio chunking
`smart_chunk_wav.py` splits WAV recordings into 1–3 second clips, using speech detection to skip long silences and prefer cuts at pauses.

- **Installation:** 

    Requires Python 3.10 or newer:
    ```
    python -m pip install numpy scipy soundfile webrtcvad-wheels
    ```
- **Usage:**
    ```
    python smart_chunk_wav.py <recordings> <chunks> --label 1
    ```
    - `recordings`: input WAV file or folder.
    - `chunks`: output folder, which must be new or empty.
    - `--label`: optional class assigned to every exported clip: `1` for allowed speakers, `0` for not allowed.
- **Output:**
    The output folder contains WAV clips, `manifest.csv` with labels and source timestamps, `skipped.csv` with reasons for skipped segments, and `run.json` with settings and statistics.

### Spectrogram creation

`create_spectrograms.py` converts audio chunks into Mel spectrograms, which are used as input for CNN models.

- **Installation:**

  Requires Python 3.10 or newer:
  ```bash
  python -m pip install numpy librosa matplotlib soundfile
  ```

- **Usage:**
  ```bash
  python m1/create_spectrograms.py spectrograms --allowed <allowed_chunks> --not-allowed <not_allowed_chunks>
  ```

  - `spectrograms`: output folder, which must be new or empty.
  - `--allowed`: folder containing allowed audio chunks and `manifest.csv`.
  - `--not-allowed`: folder containing not-allowed audio chunks and `manifest.csv`.

- **Preprocessing:**

  Audio chunks are converted to mono at 16 kHz and padded with silence to 3 seconds. Mel spectrograms are generated using 64 Mel bands, FFT size 1024 and hop length 256. The spectrograms are converted to decibels and saved as float32 arrays of shape `(64, 188)`.

- **Output:**

  The output folder contains:
  - `npy/allowed/` and `npy/not_allowed/` with spectrogram arrays saved as `.npy` files, preserving the original chunk folder structure.
  - `spectrograms.csv` with file paths, class labels and metadata from `manifest.csv`.
  - `examples/` with three example spectrograms per class saved as PDF files.
  - `run.json` with settings and the number of generated spectrograms.

  Audio recordings and chunks are stored on Google Drive. Generated spectrograms are stored locally and are not committed to the repository. Selected example PDFs can be added to GitHub.

