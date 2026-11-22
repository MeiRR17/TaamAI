"""
Torah Cantillation Dataset Pipeline
Aligns Whisper transcriptions with Sefaria's punctuated text,
extracts cantillation marks, slices audio, and organizes dataset.
"""

import requests
import json
import hashlib
import html
import math
import os
import tempfile
import logging
import re
from pathlib import Path
from pydub import AudioSegment
import regex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Cantillation mark mapping (Unicode U+0591 to U+05AF)
CANTILLATION_MAP = {
    '\u0591': 'zarka',
    '\u0592': 'munach',
    '\u0593': 'shofar_holech',
    '\u0594': 'segolta',
    '\u0595': 'pazer',
    '\u0596': 'telisha_ketana',
    '\u0597': 'telisha_gedola',
    '\u0598': 'azla',
    '\u0599': 'geresh',
    '\u059a': 'pasek',
    '\u059b': 'revia',
    '\u059c': 'shnei_gereshin',
    '\u059d': 'darga',
    '\u059e': 'tevir',
    '\u059f': 'mercha',
    '\u05a0': 'tarcha_tipcha',
    '\u05a1': 'etnachta',
    '\u05a2': 'shofar_mehuppach',
    '\u05a3': 'kadma',
    '\u05a4': 'trei_kadmin',
    '\u05a5': 'kadma_v_azla',
    '\u05a6': 'kadma_gadol',
    '\u05a7': 'shalshelet',
    '\u05a8': 'trei_taamei',
    '\u05a9': 'yativ',
    '\u05aa': 'sof_pasuk',
    '\u05ab': 'yerach_ben_yomo',
    '\u05ac': 'karne_para',
    '\u05ad': 'telisha_gedola_alt',
    '\u05ae': 'pazer_alt',
    '\u05af': 'munach_alt'
}


def fetch_sefaria_text(reference):
    """
    Fetch Hebrew text with Niqqud and Ta'amim from Sefaria API.
    
    Args:
        reference: Sefaria reference string (e.g., "Genesis.1.1-6.8")
    
    Returns:
        String containing the Hebrew text with cantillation marks
    """
    url = f"https://www.sefaria.org/api/texts/{reference}"
    params = {
        'context': 0,
        'hebr': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Sefaria ranges commonly return nested chapter/verse arrays.
        # Flatten only Hebrew text; do not accidentally align to title data.
        def flatten_hebrew(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [part for item in value for part in flatten_hebrew(item)]
            return []

        parts = flatten_hebrew(data.get('he')) if isinstance(data, dict) else []
        if not parts:
            logger.warning("Sefaria returned no Hebrew text for %s", reference)
            return None

        text = html.unescape(' '.join(parts))
        text = re.sub(r'<[^>]*>', ' ', text)
        text = ' '.join(text.split())
        if not text:
            logger.warning("Sefaria returned empty Hebrew text for %s", reference)
            return None
        
        logger.info(f"Successfully fetched text for {reference}")
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Sefaria text for {reference}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for {reference}: {e}")
        return None


def extract_cantillation(word):
    """
    Extract cantillation mark from a Hebrew word using regex.
    
    Args:
        word: Hebrew word with potential cantillation marks
    
    Returns:
        Cantillation label (folder-friendly name) or None if not found
    """
    # Regex pattern for cantillation marks (U+0591 to U+05AF)
    pattern = r'[\u0591-\u05AF]'
    
    matches = regex.findall(pattern, word)
    
    if not matches:
        return None
    
    # Return the first found cantillation mark
    cantillation_char = matches[0]
    
    # Map to folder-friendly name
    if cantillation_char in CANTILLATION_MAP:
        return CANTILLATION_MAP[cantillation_char]
    else:
        logger.warning(f"Unknown cantillation character: {cantillation_char}")
        return None


def strip_niqqud_and_cantillation(text):
    """
    Strip Niqqud (vowels) and Ta'amim (cantillation marks) from Hebrew text.
    Used for alignment comparison.
    
    Args:
        text: Hebrew text with diacritics
    
    Returns:
        Plain Hebrew text without diacritics
    """
    # Remove Niqqud (U+05B0 to U+05C7)
    text = regex.sub(r'[\u05B0-\u05C7]', '', text)
    # Remove Ta'amim (U+0591 to U+05AF)
    text = regex.sub(r'[\u0591-\u05AF]', '', text)
    # Remove other diacritics
    text = regex.sub(r'[\u05C1-\u05C2]', '', text)
    
    return text


def normalize_alignment_word(value):
    """Compare Hebrew letters only, ignoring vowels, tropes and punctuation."""
    if not isinstance(value, str):
        return ""
    plain = strip_niqqud_and_cantillation(value).lower()
    return regex.sub(r"[^\u05d0-\u05ea]", "", plain)


def align_words(whisper_word, sefaria_words, cursor, lookahead=20):
    """Match only forward from the previous aligned word.

    Repeated words can carry different cantillation marks. Searching the
    entire reference for each Whisper word would mislabel all later repeats.
    A bounded lookahead permits minor omissions while avoiding unrelated
    matches far ahead in the source text. A failed match does not move the
    cursor, so intervening noise does not shift all remaining assignments.
    """
    target = normalize_alignment_word(whisper_word)
    if not target:
        return None, cursor
    stop = min(len(sefaria_words), cursor + lookahead)
    for index in range(cursor, stop):
        if normalize_alignment_word(sefaria_words[index]) == target:
            return sefaria_words[index], index + 1
    logger.warning("Could not align word near reference index %s: %r",
                   cursor, whisper_word)
    return None, cursor


def slice_audio(audio, start_ms, end_ms, output_path):
    """Save a bounded audio slice atomically; report actual success or failure.

    The caller decodes the audio once per run. Earlier versions reopened the
    entire recording for every word and counted failed exports as successes.
    """
    output = Path(output_path)
    temporary = None
    if start_ms < 0 or start_ms >= end_ms or end_ms > len(audio):
        logger.warning("Invalid audio interval %s..%s for %s ms recording",
                       start_ms, end_ms, len(audio))
        return False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".slice-", suffix=".wav", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        audio[start_ms:end_ms].export(str(temporary), format="wav")
        if temporary.stat().st_size <= 44:
            raise ValueError("Audio slice is empty")
        os.replace(temporary, output)
        temporary = None
        logger.debug("Saved audio segment: %s", output)
        return True
    except (OSError, ValueError, Exception) as exc:
        logger.error("Failed to save audio slice %s: %s", output, exc)
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_whisper_json(whisper_json_path, audio_path, sefaria_text, root_dir):
    """
    Process Whisper JSON file, align words, extract cantillation, and slice audio.
    
    Args:
        whisper_json_path: Path to Whisper JSON file
        audio_path: Path to audio file
        sefaria_text: Sefaria text with cantillation marks
        root_dir: Root directory for output
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'total_words': 0,
        'aligned_words': 0,
        'extracted_cantillation': 0,
        'saved_segments': 0,
        'failed_alignment': 0,
        'no_cantillation': 0,
        'failed_slices': 0
    }
    
    # Load Whisper JSON
    try:
        with open(whisper_json_path, 'r', encoding='utf-8') as f:
            whisper_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Whisper JSON {whisper_json_path}: {e}")
        return stats

    if not isinstance(whisper_data, dict) or not isinstance(
        whisper_data.get("segments"), list
    ):
        logger.error("Whisper JSON must contain a segments array: %s",
                     whisper_json_path)
        return stats

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as exc:
        logger.error("Unable to decode audio %s: %s", audio_path, exc)
        return stats

    # Identify source and maintain a global word index across all Whisper
    # segments. Segment-local enumerate() silently overwrote earlier WAV files.
    base_filename = re.sub(r"[^A-Za-z0-9_-]", "_", Path(audio_path).stem)[:48] or "audio"
    source_identity = "\\n".join((
        str(Path(audio_path).resolve()),
        str(Path(whisper_json_path).resolve()),
        sefaria_text,
    ))
    source_id = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:12]
    reference_words = sefaria_text.split()
    reference_cursor = 0
    global_word_index = 0

    # Process each word segment
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            if 'words' in segment:
                for word_data in segment['words']:
                    stats['total_words'] += 1
                    global_word_index += 1
                    
                    if not isinstance(word_data, dict):
                        stats['failed_alignment'] += 1
                        continue
                    whisper_word = str(word_data.get('word') or '').strip()
                    start_time = word_data.get('start')
                    end_time = word_data.get('end')
                    
                    # Align with Sefaria text
                    sefaria_word, reference_cursor = align_words(
                        whisper_word, reference_words, reference_cursor
                    )

                    if sefaria_word:
                        stats['aligned_words'] += 1
                        
                        # Extract cantillation mark
                        cantillation = extract_cantillation(sefaria_word)
                        
                        if cantillation:
                            stats['extracted_cantillation'] += 1
                            
                            # Whisper timestamps are seconds. Reject absent,
                            # non-finite and out-of-range values rather than
                            # silently clipping or emitting mislabeled audio.
                            try:
                                start_seconds = float(start_time)
                                end_seconds = float(end_time)
                                if (not math.isfinite(start_seconds)
                                        or not math.isfinite(end_seconds)):
                                    raise ValueError("non-finite timestamp")
                                start_ms = int(start_seconds * 1000)
                                end_ms = int(end_seconds * 1000)
                            except (TypeError, ValueError, OverflowError):
                                stats['failed_slices'] += 1
                                logger.warning(
                                    "Invalid Whisper timestamps for word %s",
                                    global_word_index
                                )
                                continue

                            output_path = os.path.join(
                                root_dir, 'dataset', cantillation,
                                f"{base_filename}_{source_id}_{global_word_index:06d}.wav"
                            )
                            if slice_audio(audio, start_ms, end_ms, output_path):
                                stats['saved_segments'] += 1
                            else:
                                stats['failed_slices'] += 1
                        else:
                            stats['no_cantillation'] += 1
                    else:
                        stats['failed_alignment'] += 1
    
    return stats


def process_dataset(config_path, root_dir):
    """
    Main orchestration function to process dataset based on configuration.
    
    Args:
        config_path: Path to dataset_map.json configuration file
        root_dir: Root directory containing audio files and metadata
    """
    logger.info(f"Starting dataset processing with config: {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration {config_path}: {e}")
        return
    
    # Process each run in configuration
    total_stats = {
        'total_words': 0,
        'aligned_words': 0,
        'extracted_cantillation': 0,
        'saved_segments': 0,
        'failed_alignment': 0,
        'no_cantillation': 0,
        'failed_slices': 0
    }

    for run_name, run_config in config.items():
        logger.info(f"Processing run: {run_name}")
        
        audio_path = os.path.join(root_dir, run_config['audio_path'])
        whisper_json_path = os.path.join(root_dir, run_config['whisper_json'])
        sefaria_ref = run_config['sefaria_ref']
        
        # Check if files exist
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            continue
        
        if not os.path.exists(whisper_json_path):
            logger.error(f"Whisper JSON not found: {whisper_json_path}")
            continue
        
        # Fetch Sefaria text
        sefaria_text = fetch_sefaria_text(sefaria_ref)
        if not sefaria_text:
            logger.error(f"Failed to fetch Sefaria text for {sefaria_ref}")
            continue
        
        # Process the run
        stats = process_whisper_json(whisper_json_path, audio_path, sefaria_text, root_dir)
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
        
        logger.info(f"Completed run {run_name}: {stats}")
    
    # Log final statistics
    logger.info("=" * 50)
    logger.info("Pipeline Complete - Final Statistics:")
    logger.info(f"Total words processed: {total_stats['total_words']}")
    logger.info(f"Words aligned successfully: {total_stats['aligned_words']}")
    logger.info(f"Cantillation marks extracted: {total_stats['extracted_cantillation']}")
    logger.info(f"Audio segments saved: {total_stats['saved_segments']}")
    logger.info(f"Failed alignments: {total_stats['failed_alignment']}")
    logger.info(f"Words without cantillation: {total_stats['no_cantillation']}")
    logger.info(f"Failed audio slices: {total_stats['failed_slices']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python cantillation_pipeline.py <config_path> <root_dir>")
        print("Example: python cantillation_pipeline.py metadata/dataset_map.json /path/to/content")
        sys.exit(1)
    
    config_path = sys.argv[1]
    root_dir = sys.argv[2]
    
    process_dataset(config_path, root_dir)
