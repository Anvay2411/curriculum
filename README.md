# LMS Curriculum - Complete Learning & Development Platform

A comprehensive learning management system curriculum covering AI agents, machine learning operations, API development, audio/text processing, infrastructure, and database benchmarking.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Core Modules](#core-modules)
5. [Setup Instructions](#setup-instructions)
6. [Module Details](#module-details)

---

## Overview

This repository provides a complete learning and development platform with:

- **AI Agent Framework**: Modular framework for building intelligent agents with tool calling and reasoning
- **Machine Learning Operations**: MLFlow integration for model versioning, deployment, and evaluation
- **API & Task Models**: Ready-to-use models for audio, classification, text, translation, and summarization
- **Infrastructure**: Docker, Kubernetes, and system configuration
- **Database Tools**: HammerDB for database benchmarking
- **Load Testing**: WRK for performance testing
- **Text Generation**: RAG systems and multiple LLM provider integrations

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Linux, macOS, or Windows
- `uv` package manager (recommended)

### Initial Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Add uv to PATH
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc

# 3. Clone and navigate to curriculum
cd curriculum

# 4. Follow specific module READMEs (see sections below)
```

---

## Project Structure

```
curriculum/
├── agents/                      # AI Agent Framework
│   ├── core/                    # Framework core components
│   ├── simple_agent.py          # Learning example: Calculator agent
│   ├── agent_structure.py       # Structured response example
│   └── framework_example.py     # Multi-tool agent demo
│
├── api-tasks-models/            # ML Models & API Tasks
│   ├── audio/                   # Audio processing
│   │   ├── ASR/                 # Speech recognition (Whisper, Wav2Vec, SeamlessM4T)
│   │   ├── speech2text/         # Speech-to-text conversion
│   │   ├── speech_diarizer/     # Speaker identification
│   │   └── text2speech/         # Text-to-speech synthesis
│   ├── classification/          # Text classification tasks
│   │   ├── PII/                 # PII detection & masking
│   │   ├── profanity_masker/    # Profanity detection & filtering
│   │   ├── sentiment_classifier/# Sentiment analysis
│   │   └── Topic_modelling/     # Topic extraction
│   ├── text/                    # Text processing tasks
│   │   ├── autocomplete/        # Text autocomplete
│   │   ├── autocorrect/         # Text autocorrection
│   │   ├── MCQ/                 # Multiple choice question generation
│   │   ├── qa_generator/        # QA pair generation
│   │   └── synonym_generator/   # Synonym generation
│   ├── summarization/           # Text summarization
│   ├── table/                   # Table-based QA
│   └── translation/             # Language translation
│
├── database/                    # Database Benchmarking
│   └── hammerdb/                # HammerDB for MySQL/PostgreSQL benchmarking
│
├── infrastructure/              # Infrastructure & DevOps
│   ├── containers/              # Docker & containerization
│   │   ├── docker/              # Docker setup and Dockerfiles
│   │   └── kubernetes/          # Kubernetes configuration
│   ├── pc/                      # PC configuration
│   │   ├── full-stack/          # Full-stack setup
│   │   ├── git/                 # Git configuration
│   │   ├── linux/               # Linux setup
│   │   ├── vscode/              # VS Code configuration
│   │   └── wsl/                 # Windows Subsystem for Linux
│   └── virtualization/          # KVM virtualization
│
├── loadbalancer/                # Load Testing & Load Balancing
│   └── wrk/                     # WRK load testing tool
│
├── ml/                          # Machine Learning
│   ├── algorithms/              # ML algorithm implementations
│   │   ├── DBSCAN.ipynb         # Density-based clustering
│   │   ├── GMM.ipynb            # Gaussian Mixture Models
│   │   ├── Hierarchical.ipynb   # Hierarchical clustering
│   │   ├── KMeans.ipynb         # K-means clustering
│   │   └── Spectral_Clustering.ipynb
│   └── ops/                     # MLFlow Operations
│       ├── level0.ipynb         # Model versioning
│       ├── level1.ipynb         # Dataset versioning
│       ├── level2.ipynb         # Deployment & inference
│       ├── level3.ipynb         # Accuracy evaluation
│       └── level4.ipynb         # Advanced operations
│
└── text-generation/             # Text Generation & RAG
    ├── api/                     # LLM provider integrations
    │   ├── gemini/              # Google Gemini API
    │   ├── groq/                # Groq API
    │   ├── huggingface/         # Hugging Face models
    │   ├── krutrim/             # Krutrim API
    │   ├── mistral-ai/          # Mistral AI API
    │   ├── ollama/              # Ollama local models
    │   ├── openai/              # OpenAI API
    │   └── together-ai/         # Together AI API
    ├── rag/                     # Retrieval-Augmented Generation
    │   ├── rag00-simple-flow/   # Basic RAG implementation
    │   ├── rag01-chunking/      # Advanced chunking techniques
    │   ├── rag02-distance-metrics/
    │   └── rag03-re-ranker/
    └── serving/                 # Model serving
        ├── ollama/              # Ollama serving
        ├── tgi/                 # Text Generation Inference
        └── vllm/                # vLLM serving

```

---

## Core Modules

### 1. **AI Agent Framework** (`agents/`)
Build intelligent agents with tool calling and reasoning loops.

**Features:**
- LLM-agnostic provider interface (Groq, OpenAI, Ollama, etc.)
- Pluggable tool system
- Short-term and long-term memory management
- Structured reasoning loop
- Learning examples included

**Quick Start:**
```bash
cd agents/
pip install -r requirements.txt
python simple_agent.py              # Calculator agent example
python framework_example.py         # Multi-tool demo
```

---

### 2. **API & Task Models** (`api-tasks-models/`)
Ready-to-use models and APIs for common NLP, audio, and classification tasks.

#### Audio Processing
- **ASR (Automatic Speech Recognition)**: Whisper, Wav2Vec, SeamlessM4T
- **Speech-to-Text**: Convert audio to text
- **Speech Diarizer**: Identify speakers in audio
- **Text-to-Speech**: Generate audio from text

#### Classification
- **PII Masking**: Detect and mask personal information
- **Profanity Masker**: Detect and filter offensive content
- **Sentiment Classification**: Analyze sentiment of text
- **Topic Modeling**: Extract topics from documents

#### Text Processing
- **Autocomplete**: Predict next words
- **Autocorrect**: Fix spelling errors
- **MCQ Generation**: Create multiple-choice questions
- **QA Generator**: Generate question-answer pairs
- **Synonym Generator**: Find synonyms

#### Other
- **Summarization**: Generate text summaries
- **Table QA**: Answer questions about tables
- **Translation**: Translate between languages

**Setup Example:**
```bash
cd api-tasks-models/audio/ASR/
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python whisper_model.py
```

---

### 3. **Machine Learning Operations** (`ml/`)
Learn ML algorithms and use MLFlow for model management.

#### ML Algorithms
- K-means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)
- Spectral Clustering

#### MLFlow Operations (Progressive Learning)
- **Level 0**: Model versioning and tracking
- **Level 1**: Dataset versioning
- **Level 2**: Model deployment and inference
- **Level 3**: Accuracy evaluation
- **Level 4**: Advanced ML operations

**Quick Start:**
```bash
cd ml/ops/
pip install -r requirements.txt
mlflow ui                  # Open MLFlow dashboard
jupyter notebook level0.ipynb
```

---

### 4. **Database Benchmarking** (`database/`)
HammerDB for TPC-H benchmarking on MySQL and PostgreSQL.

**Setup:**
```bash
cd database/hammerdb/
# See mysql/ or postgresql/ README for database setup
wget https://github.com/TPC-Council/HammerDB/releases/download/v4.8/HammerDB-4.8-Linux.tar.gz
tar -zxvf HammerDB-4.8-Linux.tar.gz
./bench.sh -d <PATH> -u postgres -p postgres -db pg -port 5432
```

---

### 5. **Load Testing** (`loadbalancer/`)
WRK for HTTP load testing and performance analysis.

**Installation & Usage:**
```bash
cd loadbalancer/wrk/
sudo apt-get install build-essential libssl-dev git
git clone https://github.com/wg/wrk.git wrk
cd wrk && sudo make && sudo cp wrk /usr/local/bin
wrk -t4 -c100 -d30s http://localhost:8080
```

---

### 6. **Infrastructure & DevOps** (`infrastructure/`)
Configuration and setup for containerization, virtualization, and system setup.

**Includes:**
- Docker & Docker Compose
- Kubernetes deployment configs
- Windows Subsystem for Linux (WSL) setup
- Git configuration
- VS Code configuration
- Full-stack development setup
- KVM virtualization

---

### 7. **Text Generation & RAG** (`text-generation/`)
Multiple LLM integrations and Retrieval-Augmented Generation systems.

#### LLM Provider APIs
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini
- Mistral AI
- Groq
- Hugging Face
- Ollama (local)
- Together AI
- Krutrim

#### Retrieval-Augmented Generation (RAG)
- **rag00**: Simple RAG flow
- **rag01**: Advanced chunking techniques
- **rag02**: Distance metrics for retrieval
- **rag03**: Re-ranker integration

#### Model Serving
- Ollama
- Text Generation Inference (TGI)
- vLLM

---

## Setup Instructions

### Global Setup
```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Add to shell configuration
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc

# 3. Verify installation
uv --version
```

### Module-Specific Setup

Follow the `README.md` file in each module directory:

```bash
# AI Agents
cd agents/
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# API Models (example)
cd api-tasks-models/audio/ASR/
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# ML Operations
cd ml/ops/
pip install -r requirements.txt
mlflow ui

# Text Generation
cd text-generation/api/openai/
# Add your API key and follow module README
```

---

## Module Details

### Running Examples

#### Agent Framework
```bash
cd agents/
python simple_agent.py              # Simple calculator agent
python agent_structure.py           # Structured responses
python framework_example.py         # Multi-tool agent with reasoning
```

#### ML Algorithms
```bash
cd ml/algorithms/
jupyter notebook KMeans.ipynb       # K-means clustering example
jupyter notebook DBSCAN.ipynb       # DBSCAN clustering example
```

#### MLFlow Tracking
```bash
cd ml/ops/
mlflow ui                           # Launch dashboard on localhost:5000
jupyter notebook level0.ipynb       # Start with model versioning
```

#### Audio Processing
```bash
cd api-tasks-models/audio/ASR/
python whisper_model.py             # Whisper speech recognition
python wav2vec_model.py             # Meta Wav2Vec
python meta_seamless.py             # Meta SeamlessM4T
```

#### Text Classification
```bash
cd api-tasks-models/classification/PII/
python main.py                      # PII detection and masking
```

---

## Key Features

✅ **Modular Architecture**: Each module is independent and self-contained
✅ **Learning-Focused**: Examples and progressive complexity levels
✅ **Production-Ready**: Core components suitable for deployment
✅ **Multi-Provider Support**: Work with multiple LLM and ML platforms
✅ **Comprehensive**: Covers AI, ML, DevOps, and infrastructure
✅ **Well-Documented**: README files in each module
✅ **Easy Setup**: Standardized setup process using uv

---

## Common Commands

```bash
# List all modules
ls -la

# Setup a module
cd <module>
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run examples
python <script_name>.py
jupyter notebook <notebook_name>.ipynb

# Launch MLFlow dashboard
cd ml/ops
mlflow ui
```

---

## Contributing

When adding new modules:
1. Create a `README.md` in the module directory
2. Include setup instructions
3. Provide working examples
4. Document all dependencies in `requirements.txt`
5. Follow the existing structure and naming conventions

---

## Resources

- [AI Agent Framework Documentation](agents/README.md)
- [ML Operations Guide](ml/ops/README.md)
- [HammerDB Documentation](database/hammerdb/README.md)
- [WRK Load Testing](loadbalancer/wrk/README.md)

---

## License & Credits

Part of the LMS Curriculum project. See individual module READMEs for specific details and licensing information.

---

## Quick Navigation

| Module | Purpose | Directory |
|--------|---------|-----------|
| Agents | AI agent framework | `agents/` |
| APIs & Models | ML models and tasks | `api-tasks-models/` |
| ML Ops | Machine learning operations | `ml/` |
| Database | Performance benchmarking | `database/` |
| Infrastructure | DevOps & configuration | `infrastructure/` |
| Load Testing | Performance testing | `loadbalancer/` |
| Text Generation | LLM integration & RAG | `text-generation/` |

---

**Last Updated**: January 2026
