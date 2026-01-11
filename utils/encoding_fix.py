"""
CSV Encoding Converter

This module provides utilities for detecting and converting CSV file encodings.
It helps fix encoding issues by converting files to UTF-8.
"""

import csv
import os
from typing import Optional

import chardet


def detect_encoding(file_path: str) -> Optional[str]:
    """
    Detect the encoding of a file using the chardet library.

    Note: This uses statistical analysis and is not 100% accurate.

    Args:
        file_path: Path to the file to detect encoding

    Returns:
        Detected encoding name (e.g., 'utf-8', 'latin-1'), or None if detection fails
    """
    try:
        with open(file_path, 'rb') as f:
            # Read first 32KB for detection
            raw_data = f.read(32 * 1024)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']

            print(f"File '{os.path.basename(file_path)}' detected encoding: '{encoding}' (confidence: {confidence:.0%})")
            return encoding
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None


def convert_csv_to_utf8(
    source_path: str,
    destination_path: str,
    source_encoding: str = 'autodetect'
) -> bool:
    """
    Convert a CSV file from one encoding to UTF-8.

    Args:
        source_path: Path to the source CSV file
        destination_path: Path to save the UTF-8 encoded CSV file
        source_encoding: Source file encoding. Use 'autodetect' to auto-detect,
                        or specify manually (e.g., 'latin-1', 'big5', 'cp1252')

    Returns:
        True if conversion succeeded, False otherwise
    """
    print("-" * 50)

    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return False

    # Auto-detect encoding if needed
    if source_encoding == 'autodetect':
        detected_encoding = detect_encoding(source_path)
        if not detected_encoding:
            print("Unable to detect encoding. Conversion cancelled.")
            return False

        # chardet sometimes misidentifies 'ascii', but 'latin-1' is more inclusive
        source_encoding = 'latin-1' if detected_encoding == 'ascii' else detected_encoding

    print(f"Attempting to read from '{source_path}' (encoding: {source_encoding})...")

    try:
        # Read source file
        with open(source_path, mode='r', encoding=source_encoding, newline='', errors='replace') as infile:
            # Write to destination file
            with open(destination_path, mode='w', encoding='utf-8', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                # Process line by line to handle large files
                for row in reader:
                    writer.writerow(row)

        print(f"Success! File converted and saved to '{destination_path}' (encoding: utf-8)")
        return True

    except FileNotFoundError:
        print(f"Error: Source file not found: {source_path}")
        return False
    except UnicodeDecodeError as e:
        print(f"Read error: Failed to read '{source_path}' using '{source_encoding}' encoding.")
        print(f"Error message: {e}")
        print("Suggestion: Try manually specifying a different encoding (e.g., 'big5', 'cp950', 'cp1252').")
        return False
    except Exception as e:
        print(f"Unknown error occurred: {e}")
        return False


def batch_convert_files(
    files_list: list[str],
    source_directory: str = './Dataset/',
    output_directory: str = './Dataset/utf8/',
    source_encoding: str = 'autodetect'
) -> dict[str, bool]:
    """
    Batch convert multiple CSV files to UTF-8.

    Args:
        files_list: List of filenames to convert
        source_directory: Directory containing source files
        output_directory: Directory to save converted files
        source_encoding: Source encoding ('autodetect' or specific encoding)

    Returns:
        Dictionary mapping filenames to conversion success (True/False)
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: '{output_directory}'")

    results = {}

    for filename in files_list:
        source_file_path = os.path.join(source_directory, filename)

        # Create output filename (e.g., 'data.csv' -> 'data_utf8.csv')
        base, ext = os.path.splitext(filename)
        destination_file_path = os.path.join(output_directory, f"{base}_utf8{ext}")

        # Convert the file
        success = convert_csv_to_utf8(
            source_file_path,
            destination_file_path,
            source_encoding=source_encoding
        )

        results[filename] = success

    return results


if __name__ == '__main__':
    """
    Example usage of the encoding converter.

    Modify the configuration below to suit your needs.
    """

    # Configuration
    files_to_process = [
        'llm_reprocess.csv',
        # Add more files here as needed
    ]

    source_directory = './Dataset/'
    output_directory = './Dataset/utf8/'

    print("=" * 60)
    print("CSV Encoding Converter - Batch Processing")
    print("=" * 60)
    print()

    # Batch convert all files
    results = batch_convert_files(
        files_list=files_to_process,
        source_directory=source_directory,
        output_directory=output_directory,
        source_encoding='autodetect'  # or specify encoding like 'latin-1'
    )

    # Print summary
    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)

    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed files:")
        for filename, success in results.items():
            if not success:
                print(f"  - {filename}")
