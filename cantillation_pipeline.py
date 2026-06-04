"""
Torah Cantillation Dataset Pipeline
Aligns Whisper transcriptions with Sefaria's punctuated text,
extracts cantillation marks, slices audio, and organizes dataset.
"""

import requests
import json
import os
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
        
        # Extract Hebrew text - handle different response structures
        if 'he' in data:
            text = data['he']
        elif 'heTitle' in data:
            # Fallback for different API response structure
            text = data.get('he', '')
        else:
            logger.warning(f"Unexpected API response structure for {reference}")
            return None
        
        # Clean text - remove extra whitespace and newlines
        text = ' '.join(text.split())
        
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


def align_words(whisper_word, sefaria_text):
    """
    Align a Whisper word to the corresponding word in Sefaria text.
    
    Args:
        whisper_word: Word from Whisper transcription
        sefaria_text: Full text from Sefaria with cantillation marks
    
    Returns:
        Matched Sefaria word with full cantillation marks, or None if no match
    """
    # Clean both words for comparison
    whisper_clean = strip_niqqud_and_cantillation(whisper_word.lower())
    
    # Split Sefaria text into words
    sefaria_words = sefaria_text.split()
    
    # Try to find matching word
    for sefaria_word in sefaria_words:
        sefaria_clean = strip_niqqud_and_cantillation(sefaria_word.lower())
        
        # Compare cleaned versions
        if whisper_clean == sefaria_clean:
            return sefaria_word
    
    # If no exact match, try fuzzy matching (remove spaces, etc.)
    whisper_fuzzy = whisper_clean.replace(' ', '').replace('-', '')
    
    for sefaria_word in sefaria_words:
        sefaria_clean = strip_niqqud_and_cantillation(sefaria_word.lower())
        sefaria_fuzzy = sefaria_clean.replace(' ', '').replace('-', '')
        
        if whisper_fuzzy == sefaria_fuzzy:
            return sefaria_word
    
    logger.warning(f"Could not align word: '{whisper_word}'")
    return None


def slice_audio(audio_path, start_ms, end_ms, output_path):
    """
    Slice audio file and save segment.
    
    Args:
        audio_path: Path to input audio file
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        output_path: Path to save sliced audio
    """
    try:
        # Load audio
        audio = AudioSegment.from_wav(audio_path)
        
        # Slice audio
        segment = audio[start_ms:end_ms]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save segment
        segment.export(output_path, format='wav')
        
        logger.debug(f"Saved audio segment: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to slice audio {audio_path}: {e}")


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
        'no_cantillation': 0
    }
    
    # Load Whisper JSON
    try:
        with open(whisper_json_path, 'r', encoding='utf-8') as f:
            whisper_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Whisper JSON {whisper_json_path}: {e}")
        return stats
    
    # Extract base filename for output naming
    base_filename = Path(audio_path).stem
    
    # Process each word segment
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            if 'words' in segment:
                for word_idx, word_data in enumerate(segment['words']):
                    stats['total_words'] += 1
                    
                    whisper_word = word_data.get('word', '').strip()
                    start_time = word_data.get('start', 0)
                    end_time = word_data.get('end', 0)
                    
                    # Align with Sefaria text
                    sefaria_word = align_words(whisper_word, sefaria_text)
                    
                    if sefaria_word:
                        stats['aligned_words'] += 1
                        
                        # Extract cantillation mark
                        cantillation = extract_cantillation(sefaria_word)
                        
                        if cantillation:
                            stats['extracted_cantillation'] += 1
                            
                            # Convert seconds to milliseconds
                            start_ms = int(start_time * 1000)
                            end_ms = int(end_time * 1000)
                            
                            # Create output path
                            output_path = os.path.join(
                                root_dir,
                                'dataset',
                                cantillation,
                                f"{base_filename}_{word_idx:03d}.wav"
                            )
                            
                            # Slice and save audio
                            slice_audio(audio_path, start_ms, end_ms, output_path)
                            stats['saved_segments'] += 1
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
        'no_cantillation': 0
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
