# content-moderation
# Text Content Moderation

A machine learning project focused on identifying toxic and potentially harmful text using traditional machine learning and transformer-based models. The repository contains two modules: binary toxicity prediction and multiclass hate-speech classification.

## Projects

### 1. Binary Toxicity Prediction

The `binary_toxicity` module classifies text comments into two categories:

- **Non-toxic**
- **Toxic**

It includes a baseline classification notebook, a fine-tuned model notebook, model configuration files, and evaluation results.

**Key components**
- Baseline model implementation
- Fine-tuned DistilBERT implementation
- Classification report and final evaluation metrics
- Performance visualizations for model comparison, confusion matrix, ROC curve, precision-recall curve, and training progress

### 2. Multiclass Hate-Speech Classification

The `multiclass_hate_speech` module focuses on classifying text into multiple predefined categories of harmful or non-harmful content.

It includes a Jupyter Notebook and an artifacts directory containing the selected model-related files.

## Repository Structure

```text
content-moderation/
│
├── binary_toxicity/
│   ├── finetunned_model/
│   │   ├── src/
│   │   └── message_violation_detection.ipynb
│   │
│   ├── models/
│   │   ├── distilbert-toxic-final/
│   │   └── message_violation_detection_baseline_model.ipynb
│   │
│   └── results/
│       ├── figures/
│       │   ├── baseline_vs_finetuned.png
│       │   ├── class_distribution.png
│       │   ├── confusion_matrix.png
│       │   ├── precision_recall_curve.png
│       │   ├── roc_curve.png
│       │   ├── threshold_performance.png
│       │   ├── training_validation_loss.png
│       │   └── validation_f1.png
│       │
│       ├── classification_report.txt
│       └── final_metrics.json
│
├── multiclass_hate_speech/
│   ├── artifacts/
│   └── text_model.ipynb
│
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook
- Matplotlib and Seaborn

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ajju170207/content-moderation.git
cd content-moderation
```

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies required by the notebook you want to run. A `requirements.txt` file is not currently included in the repository.

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook for the module you want to explore:

**Binary toxicity prediction**
- `binary_toxicity/finetunned_model/message_violation_detection.ipynb`
- `binary_toxicity/models/message_violation_detection_baseline_model.ipynb`

**Multiclass hate-speech classification**
- `multiclass_hate_speech/text_model.ipynb`

Running the complete workflows may require separately obtaining the datasets and trained model weights, which are not included in the repository.

## Evaluation

The binary toxicity module includes saved evaluation reports and visualizations to examine model performance, classification errors, precision-recall trade-offs, and training progress.

The multiclass module provides a notebook for exploring the configured multiclass classification workflow.
