# Agentic-News-Bot

A personalized press agent powered by AI, featuring news generation, fake news detection, and press conference simulation capabilities.

## 🎯 Key Features

- **News Generation**: Automated news creation
- **Fake News Detection**: ML-powered detection to identify unreliable news articles
- **Press Conference Simulator**: Interactive press conference simulation system

> **Note**: At the moment This repository currently contains the complete fake news detection implementation. Other features (news generation and press conference simulator) are yet to be integrated .

## 📁 Project Structure

```
Agentic-News-Bot/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not tracked)
├── .gitignore                      # Git ignore rules
│
├── architecture/                   # Project setup and documentation
│   └── project-structure-script.sh # Script to generate project structure
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── fake-news-detection.ipynb  # Fake news detection analysis
│
├── src/                            # Source code
│   ├── agents/                     # AI agents
│   │   └── news_prediction_agent.py
│   │
│   ├── data/                       # Datasets
│   │   └── News_dataset/
│   │       ├── Fake.csv            # Fake news samples
│   │       └── True.csv            # True news samples
│   │
│   ├── embeddings/                 # Text embedding models
│   │   └── embed_model.py
│   │
│   └── models/                     # Trained ML models
│       ├── best_model.pkl          # Best performing model
│       ├── logisticRegressor.pkl   # Logistic regression model
│       ├── minmax_scaler.pkl       # Feature scaler
│       ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│       └── embedding_model/        # Pre-trained sentence transformer
│
├── templates/                      # HTML templates
│   └── index.html                  # Web interface
│
├── tests/                          # Test files
│   └── news_prediction.py          # Prediction tests
│
└── utils/                          # Utility functions
    ├── data_preprocessing.py       # Data cleaning and preprocessing
    ├── data_validation.py          # Input validation
    ├── simulation_helpers.py       # Simulation utilities
    └── train_and_save_model.py     # Model training pipeline
```

### Quick Setup

To recreate the project structure from scratch, run:

```bash
bash architecture/project-structure-script.sh
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/IyedGuezmir/Agentic-News-Bot.git
cd Agentic-News-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🧠 Fake News Detection

The fake news detection system uses a hybrid approach:

### Agent Workflow
1. **Text Embedding**: News articles are converted to semantic embeddings using `all-MiniLM-L6-v2` Sentence Transformer
2. **ML Prediction**: Pre-trained classifier predicts if the news is fake or true with confidence score
3. **Web Verification**: LLM (GPT-4) with web search tools verifies the news against credible online sources
4. **Final Decision**: If web verification finds credible sources, marks as True News; otherwise, defers to ML model prediction

### Key Components
- **Sentence Transformers** (`all-MiniLM-L6-v2`): For semantic text embeddings
- **Pre-trained ML Classifier**: For initial prediction
- **LangChain + OpenAI GPT-4**: For intelligent web-based verification
- **Hybrid Decision Logic**: Combines ML predictions with real-time web verification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
