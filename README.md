# Torah Cantillation Dataset Pipeline

A Python pipeline to automatically align Whisper transcriptions with Sefaria's punctuated Torah text, extract cantillation marks (Ta'amei HaMikra), slice audio segments, and organize them into a dataset structure for deep learning classification.

## Features

- **Sefaria API Integration**: Fetches Hebrew text with Niqqud and Ta'amim from Sefaria
- **Cantillation Mark Detection**: Regex-based extraction of 26 cantillation marks (Unicode U+0591 to U+05AF)
- **Robust Text Alignment**: Handles spelling differences between Whisper and Sefaria
- **Audio Slicing**: Uses pydub to slice audio based on Whisper timestamps
- **Configuration-Driven**: Batch processing via JSON configuration file
- **Comprehensive Logging**: Tracks progress and errors with detailed statistics

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Note: pydub requires FFmpeg to be installed on your system. Install it from https://ffmpeg.org/

## Project Structure

```
c:\MR Project\TaamAI\
├── cantillation_pipeline.py    # Main pipeline script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── pipeline.log                 # Generated log file
└── metadata\
    └── dataset_map.json         # Configuration file (create this)
```

## Configuration

Create a `metadata/dataset_map.json` file to define your processing runs:

```json
{
  "run_1": {
    "audio_path": "Reader_1_Dvir_Porati/Genesis.wav",
    "whisper_json": "metadata/Genesis_whisper.json",
    "sefaria_ref": "Genesis.1.1-6.8"
  },
  "run_2": {
    "audio_path": "Reader_1_Dvir_Porati/Exodus.wav",
    "whisper_json": "metadata/Exodus_whisper.json",
    "sefaria_ref": "Exodus.1.1-5.10"
  }
}
```

**Configuration Fields:**
- `audio_path`: Relative path to the audio file from root directory
- `whisper_json`: Relative path to the Whisper JSON transcription file
- `sefaria_ref`: Sefaria reference string (e.g., "Genesis.1.1-6.8")

## Usage

Run the pipeline with the configuration file and root directory:

```bash
python cantillation_pipeline.py metadata/dataset_map.json /path/to/content
```

**Example:**
```bash
python cantillation_pipeline.py metadata/dataset_map.json "/path/to/Google_Drive/Academia&Research/כריית מידע/Deep Learning/Project/content"
```

## Output Structure

The pipeline creates a dataset organized by cantillation marks:

```
dataset/
├── zarka/
│   ├── Genesis_000.wav
│   ├── Genesis_001.wav
│   └── ...
├── munach/
│   ├── Genesis_005.wav
│   └── ...
├── segolta/
│   └── ...
└── ...
```

**Output Naming Convention:** `{original_filename}_{word_index:03d}.wav`

## Cantillation Classes

The pipeline detects and categorizes the following 26 cantillation marks:

- zarka
- munach
- shofar_holech
- segolta
- pazer
- telisha_ketana
- telisha_gedola
- azla
- geresh
- pasek
- revia
- shnei_gereshin
- darga
- tevir
- mercha
- tarcha_tipcha
- etnachta
- shofar_mehuppach
- kadma
- trei_kadmin
- kadma_v_azla
- kadma_gadol
- shalshelet
- trei_taamei
- yativ
- sof_pasuk

## Logging

The pipeline generates a `pipeline.log` file with detailed information:
- API fetch results
- Alignment successes/failures
- Cantillation extraction results
- Audio slicing operations
- Final statistics

## Error Handling

The pipeline includes robust error handling:
- Skips words without recognizable cantillation marks
- Logs alignment mismatches without crashing
- Gracefully handles API failures
- Continues processing even if individual words fail

## Statistics

After completion, the pipeline logs:
- Total words processed
- Words aligned successfully
- Cantillation marks extracted
- Audio segments saved
- Failed alignments
- Words without cantillation

## Requirements

- Python 3.7+
- requests>=2.31.0
- pydub>=0.25.1
- regex>=2023.10.3
- FFmpeg (for pydub audio processing)

## Troubleshooting

**FFmpeg not found error:**
- Install FFmpeg from https://ffmpeg.org/
- Ensure FFmpeg is in your system PATH

**API timeout:**
- Check your internet connection
- The script has a 30-second timeout for API calls

**Alignment failures:**
- Check that Whisper transcription matches the Sefaria reference
- Review the log file for specific word alignment issues

**Audio slicing errors:**
- Ensure audio files are in WAV format
- Verify that Whisper timestamps are valid

## License

This project is part of the TaamAI dataset creation initiative.
