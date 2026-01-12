# streamlit_ui/api_service.py

import streamlit as st
import requests
import json
from streamlit_config import PREDICT_ENDPOINT, CORRECTION_ENDPOINT, API_BASE_URL

def predict_fen(image_file):
    """Sends an image to the API for FEN prediction."""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(PREDICT_ENDPOINT, files=files)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the API: {e}")
        st.warning("Please ensure the API service is running and the API_BASE_URL is correct.")
        return None

def submit_correction(prediction_id, corrected_fen):
    """Sends a corrected FEN back to the API for model training."""
    try:
        correction_payload = {
            "prediction_id": prediction_id,
            "corrected_fen": corrected_fen
        }
        response = requests.post(CORRECTION_ENDPOINT, json=correction_payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while submitting correction: {e}")
        return None

def switch_service(service_type):
    """Switches between end-to-end model and YOLO-pipeline."""
    try:
        switch_payload = {
            "service_type": service_type
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(f"{API_BASE_URL}/service/switch", json=switch_payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while switching service: {e}")
        return None