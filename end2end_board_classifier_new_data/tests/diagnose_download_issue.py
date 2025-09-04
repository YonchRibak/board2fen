import requests
import sys
import os
import logging
from pathlib import Path

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GCS_ANNOTATIONS_URL = "https://storage.googleapis.com/chess_red_dataset/annotations.json"
LOCAL_CACHE_DIR = Path("./gcs_cache")
LOCAL_ANNOTATIONS_PATH = LOCAL_CACHE_DIR / "annotations.json"


def diagnose_download():
    """Diagnoses network issues preventing the annotations download."""
    logger.info("Starting network diagnostics for GCS download.")

    # Check for file existence first, as it's the fastest check.
    if LOCAL_ANNOTATIONS_PATH.exists():
        logger.info(
            f"File already exists at {LOCAL_ANNOTATIONS_PATH}. Size: {os.path.getsize(LOCAL_ANNOTATIONS_PATH) / 1024 / 1024:.2f}MB")
        logger.info("No download needed. Exiting.")
        return

    try:
        # A quick check to see if the URL is reachable at all
        logger.info(f"Attempting to connect to {GCS_ANNOTATIONS_URL}...")
        response = requests.head(GCS_ANNOTATIONS_URL, timeout=10)
        response.raise_for_status()
        logger.info("Connection successful.")

        # Now try to download a small chunk to confirm data can be streamed
        logger.info("Attempting to download a small chunk of the file...")
        response = requests.get(GCS_ANNOTATIONS_URL, stream=True, timeout=10)
        with open(LOCAL_ANNOTATIONS_PATH, 'wb') as f:
            chunk = next(response.iter_content(chunk_size=1024))
            if chunk:
                f.write(chunk)

        logger.info("Successfully downloaded a chunk.")
        logger.info(
            f"Please run the main test script again. If the issue persists, your network may be blocking large files.")

    except requests.exceptions.Timeout:
        logger.error("Request timed out. This often indicates a slow or blocked connection to the server.")
        logger.error("Check your firewall or network settings. You may be behind a restrictive proxy.")
    except requests.exceptions.RequestException as e:
        logger.error(f"A general request error occurred: {e}")
        logger.error("This could be due to a DNS issue, an active firewall, or an incorrect proxy configuration.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during diagnostics: {e}")
    finally:
        # Clean up the partial file if it was created
        if LOCAL_ANNOTATIONS_PATH.exists() and os.path.getsize(LOCAL_ANNOTATIONS_PATH) < 10240:
            os.remove(LOCAL_ANNOTATIONS_PATH)
            logger.info("Cleaned up partial download file.")


if __name__ == "__main__":
    diagnose_download()
