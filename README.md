# Chess Position Recognition: End-to-End vs Pipeline Approach

A comparative study exploring different computer vision approaches for chess position recognition in challenging real-world conditions.

## Project Overview

This project investigates the effectiveness of different image recognition approaches when applied to complex spatial recognition tasks. Using chess position recognition as a test case, we compare two fundamentally different methodologies:

1. **End-to-End CNN Model**: A single neural network trained directly on chess position matrices
2. **Pipeline Approach**: A multi-stage system using YOLO models and image transformation techniques

Chess position recognition serves as an ideal benchmark for this comparison because it requires:
- Detection and classification of multiple different piece types
- Understanding of spatial relationships and orientation
- Accurate placement of pieces on a structured grid
- Handling of various lighting conditions, angles, and chess set designs

## Dataset: Chess Recognition Dataset (ChessReD)

This project utilizes the Chess Recognition Dataset (ChessReD), a comprehensive collection of real-world chess position images designed specifically for chess recognition research.

### Dataset Overview

ChessReD comprises a diverse collection of 10,800 images of chess formations captured using smartphone cameras, ensuring real-world applicability. The dataset was methodically constructed to provide maximum variability in chess positions and imaging conditions.

### Data Collection Methodology

The dataset collection process was designed to ensure comprehensive coverage of chess positions:

- **Position Diversity**: Used the Encyclopedia of Chess Openings (ECO) classification system, selecting 20 ECO codes from each of the five volumes (100 subcategories total)
- **Game Selection**: Each ECO code was matched to actual played chess games following those opening sequences, creating a set of 100 complete games
- **Image Capture**: Games were played out on a physical chessboard with images captured after each move using Portable Game Notation (PGN) data

### Technical Specifications

**Hardware Diversity:**
- Three distinct smartphone models: Apple iPhone 12, Huawei P40 Pro, Samsung Galaxy S8
- Different camera specifications and sensor types for increased dataset robustness
- Resolution variations: 3072×3072 (Huawei) and 3024×3024 (Apple/Samsung)

**Imaging Conditions:**
- Diverse viewing angles: top-view to oblique angles
- Multiple perspectives: white player view, side view, arbitrary bystander positions
- Varied lighting conditions: natural and artificial light sources
- Real-world scenarios simulating practical usage conditions

### Annotations and Labels

**Position Annotations:**
- 12 piece categories (6 piece types × 2 colors)
- Chessboard coordinates in algebraic notation (e.g., "a8")
- Automatic extraction from Forsyth-Edwards Notation (FEN) strings
- Complete position state for every image in the dataset

**Additional Annotations (Subset):**
- Bounding box annotations for individual pieces
- Chessboard corner coordinates for 20 selected games
- Corner classification based on white player's perspective
- Orientation and perspective information

### Dataset Splits

**Main Dataset (10,800 images):**
- Training: 6,479 images (60%)
- Validation: 2,192 images (20%)
- Test: 2,129 images (20%)
- Game-level splitting to prevent data leakage between similar positions
- Stratified by smartphone camera type

**Annotated Subset (2,078 images):**
- Training: 1,442 images (14 games)
- Validation: 330 images (3 games)
- Test: 306 images (3 games)
- 70/15/15 split with camera stratification

## Methodology

### End-to-End Approach

The end-to-end model treats chess recognition as a direct image-to-position mapping problem. A convolutional neural network is trained to output a complete chess position matrix from a single input image.

**Advantages:**
- Superior accuracy according to research findings (as demonstrated in the referenced thesis work)
- Simplified data labeling - only requires chess position annotations (FEN notation or position matrix)
- Easier integration with crowdsourced data collection
- Holistic understanding of the chess board state

**Challenges:**
- Resource-intensive training requiring significant GPU memory and time
- Requires substantial training data for robust performance
- Black-box approach with limited interpretability

### Pipeline Approach

The pipeline method breaks down the problem into sequential, specialized tasks:
1. Chess board detection and corner identification using YOLO
2. Perspective correction through image warping
3. Individual piece detection and classification using a second YOLO model

**Advantages:**
- More interpretable and debuggable components
- Can leverage existing object detection frameworks
- Potentially faster inference time

**Challenges:**
- Error accumulation across pipeline stages
- Complex data labeling requirements (bounding boxes, corner coordinates)
- Difficult to fine-tune end-to-end
- More complex system architecture

## Interactive Demonstration

A Streamlit-based web interface allows users to:
- Upload chess position images
- Toggle between end-to-end and pipeline recognition services
- Compare results side-by-side
- Manually correct predictions using an interactive chess board editor
- Submit corrected positions to improve the training dataset

## Crowdsourced Data Collection

The system implements a feedback loop for continuous improvement:

1. Users upload chess position images
2. Both models provide predictions
3. Users can correct inaccuracies using the interactive board interface
4. Corrected positions are stored with FEN notation
5. Every 1,000 new entries trigger automated fine-tuning of the end-to-end model

This approach particularly benefits the end-to-end model, as it only requires position labels rather than the complex bounding box and corner annotations needed for the pipeline approach.

## Technical Implementation

### Notebooks

- **[End-to-End Approach Notebook](./notebooks/end_to_end_approach.ipynb)** - Implementation and training of the CNN-based position recognition model
- **[Pipeline Approach Notebook](./notebooks/pipeline_approach.ipynb)** - Implementation of the YOLO-based detection and warping pipeline

### Data Format

The system uses FEN (Forsyth-Edwards Notation) as the standard format for chess positions, which can be easily converted to/from position matrices for training and evaluation.

## Research Context

This work builds upon recent research in end-to-end chess recognition, particularly the findings from "End-to-End Chess Recognition" by Athanasios Masouris (2023), which demonstrated superior performance of direct CNN approaches compared to traditional pipeline methods on the ChessReD dataset.

## Future Work

- Automated model retraining pipeline
- React Native app for easier user interaction.


---

## About the Author

This project was developed by **Jonathan Ribak**, a fullstack developer and data scientist, as a final project for the Data Science course at John Bryce Academy.

Jonathan combines expertise in software development with machine learning and computer vision techniques to explore practical applications of AI in real-world scenarios. This chess position recognition project demonstrates the integration of modern deep learning approaches with traditional computer vision pipelines, showcasing both the technical challenges and practical considerations involved in deploying machine learning systems.

**Background:**
- Fullstack Developer with experience in end-to-end application development
- Data Scientist specializing in computer vision and machine learning applications
- Student at John Bryce Academy's Data Science program

**Project Focus:**
This work represents an exploration of comparative methodologies in computer vision, specifically examining the trade-offs between end-to-end deep learning approaches and traditional pipeline-based systems in challenging real-world recognition tasks.

---

## Local Setup Instructions

### Prerequisites

Install all required packages using the requirements file:

```bash
pip install -r requirements-full.txt
```

**Note:** This installation may take considerable time due to large packages including PyTorch, TensorFlow, and Ultralytics YOLO dependencies.

### API Setup

1. Ensure Docker Desktop is running on your system
2. Build and start the API services:

```bash
docker compose up --build
```

**Warning:** This build process will take an extended period (potentially 30+ minutes) due to the installation of large machine learning packages including:
- PyTorch
- TensorFlow
- Ultralytics (YOLO)

### Streamlit UI Setup

The UI requires two separate terminal sessions running simultaneously:

#### Terminal 1: Board Editor Component

Navigate to the custom component directory and start the development server:

```bash
cd streamlit_ui/my_component/frontend
npm install
npm run start
```

This will start the React-based chess board editor component.

#### Terminal 2: Streamlit Application

In a separate terminal, launch the main Streamlit application:

```bash
streamlit run streamlit_ui/app.py
```

### Running the Application

1. Ensure Docker services are running (API setup complete)
2. Verify the board editor component is running (Terminal 1)
3. Confirm the Streamlit app is running (Terminal 2)
4. Open your browser to the Streamlit app URL (typically `http://localhost:8501`)
5. Upload chess position images and compare the two recognition approaches

### Troubleshooting

- If Docker build fails, ensure sufficient disk space (several GB required for ML packages)
- If the board editor component doesn't load, verify Node.js dependencies are installed in the frontend directory
- If API connections fail, check that Docker containers are healthy and ports are not conflicted