> **Script consolidation (2026-11-03):** Selected utility scripts from this repository were integrated into [Learning_Modules/tools/taam-cantillation](https://github.com/MeiRR17/Learning_Modules/tree/main/tools/taam-cantillation), via [PR #4](https://github.com/MeiRR17/Learning_Modules/pull/4). The import includes source provenance and CI syntax/content checks. The original repository remains accessible, may contain additional data or configuration, and has **not been archived or deleted**. Consult the consolidated README for exclusions and portability changes.

---

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

**Output Naming Convention (feature branch):** `{audio_stem}_{source_id}_{global_word_index:06d}.wav`. A deterministic 12-character source fingerprint and a word index spanning all Whisper segments prevent collisions that previously overwrote samples when a new segment restarted its local word count.

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


## Ordered alignment and reliable slices (feature branch)

The previous alignment code scanned the entire Sefaria reference independently for each Whisper word and selected the first matching spelling. Repeated Hebrew words can carry different cantillation marks, so this systematically assigned the wrong mark to later occurrences. Alignment now advances a **monotonic cursor** through the Sefaria reference, considers up to 20 words ahead and compares Hebrew letters after removing niqqud, trope marks and punctuation. When a word does not match in that window, it is marked unaligned rather than assigning a distant, potentially incorrect accent. The algorithm is conservative and is **not** a verified forced aligner: the generated labels still require review for research use.

Sefaria API responses can contain nested chapter/verse arrays and HTML markup. Those responses are flattened and cleaned before alignment. A single audio decode is shared by all word slices for that Whisper file rather than reopening the recording for every word. Word timestamps must be finite, ordered and within the decoded audio duration. Invalid timestamps and failed exports are tracked as `failed_slices`; `saved_segments` counts only WAV files actually written. Segments are written to temporary files and promoted into the dataset folder only after successful export.

Because output filenames changed to avoid overwrites, a new run does **not** automatically delete WAVs generated by the old naming scheme. For a clean dataset, choose a new output root or manually archive the legacy dataset before rerunning the upgraded pipeline. Do not mix labels from the old and new alignment algorithms without distinguishing their provenance.

The Sefaria API is still required for each run; the repository contains no built-in offline Torah reference dataset. This PR is code-only and has not been tested or merged.
