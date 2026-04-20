# Fake News Detection with FDHN (Fuzzy Deep Hybrid Network)

A deep learning system that detects fake news using Fuzzy Deep Hybrid Networks with multi-modal input processing.

## What is this project?

This project implements an Enhanced Fake News Detection System using **FDHN (Fuzzy Deep Hybrid Network)** architecture that processes multiple types of information:

- **News Statement Text** - The actual claim being fact-checked
- **Contextual Information** - Speaker, subject, and context details  
- **Numerical Features** - Historical credibility scores and metadata
- **Justification Text** - Detailed reasoning (LIAR2 only)
- **Fuzzy Logic** - Handles uncertainty in classification

### Models Available:
- **LIAR Model** (4-module): Basic fake news detection
- **LIAR2 Model** (5-module): Enhanced with justification reasoning

### Classification Labels:
- **LIAR**: true, mostly-true, half-true, barely-true, false, pants-on-fire
- **LIAR2**: 0-5 (0=pants-on-fire, 1=barely-true, 2=false, 3=half-true, 4=mostly-true, 5=true)

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train LIAR model (4-module)
python train_liar.py

# Train LIAR2 model (5-module) 
python train_liar2.py
```

### 3. Run Web Interface
```bash
# Start backend server
cd backend
python app.py



### 4. Access Application
- Open `http://localhost:5000` in your browser
- Select model type (LIAR or LIAR2)
- Input statement details
- Get fake news prediction with confidence scores

## Project Structure
```
├── data/                 # Training datasets
│   ├── LIAR/            # Original LIAR dataset  
│   └── liar2/           # Enhanced LIAR2 dataset
├── models/              # Trained model files
├── src/
│   ├── models/          # FDHN model architectures
│   └── data/            # Data processing utilities
├── frontend/            # Web interface
├── backend/             # Flask API server
└── Examples.md          # Test examples
```

## Quick Test
Use examples from `Examples.md` to test both models and compare their performance differences.