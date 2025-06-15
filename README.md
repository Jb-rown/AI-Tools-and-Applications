# AI Tools Application: Mastering the AI Toolkit

## Overview
The **AI Tools Assignment** is a  project demonstrating proficiency in AI tools and frameworks, including Scikit-learn, TensorFlow, and spaCy, applied to real-world machine learning and NLP tasks. This project showcases theoretical understanding, practical implementation, and ethical considerations in AI development. The assignment aligns with the theme "Mastering the AI Toolkit" and includes tasks for classical ML, deep learning, NLP, and a bonus deployment using Streamlit.

## Project Objectives
- **Theoretical Understanding**: Analyze and compare AI tools (TensorFlow, PyTorch, Scikit-learn, spaCy) for their applications, ease of use, and community support.
- **Practical Implementation**: Build and evaluate models for:
  - Classical ML with the Iris Species Dataset.
  - Deep learning with the MNIST Handwritten Digits Dataset.
  - NLP with Amazon Product Reviews for named entity recognition (NER) and sentiment analysis.
- **Ethics & Optimization**: Address biases, mitigate ethical concerns, and debug TensorFlow code.
- **Bonus Task**: Deploy the MNIST classifier as a web app using Streamlit.


## Project Structure
```
AI-Tools-Applications/                     
├── iri.ipynb   # Scikit-learn decision tree
├── Handwritten_digits.py              # TensorFlow CNN for MNIST
├── name_entity_recognition.py.py      # spaCy for NER and sentiment analys
├── mnist_streamlit.py        # Streamlit app code
├── mnist_cnn.h5             # Pre-trained MNIST model
├── AI_Tools_Application_Report.pdf  # Final report
├── README.md                     # Project documentation
```

## Technologies Used
- **Frameworks**: Scikit-learn, TensorFlow, spaCy
- **Platforms**: Google Colab (for GPU support), Jupyter Notebook
- **Datasets**: Iris Species (Kaggle), MNIST Handwritten Digits (TensorFlow Datasets), Amazon Product Reviews
- **Deployment**: Streamlit
- **Languages**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, TextBlob (for sentiment analysis)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[Your-Repo]/AI-Tools-Assignment.git
   cd AI-Tools-Application
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Create a `requirements.txt` with:
   ```
   scikit-learn==1.2.2
   tensorflow==2.12.0
   spacy==3.5.0
   textblob==0.17.1
   streamlit==1.22.0
   pandas==2.0.0
   numpy==1.24.3
   matplotlib==3.7.1
   ```
   Install spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run Notebooks**:
   - Open `notebooks/` in Jupyter Notebook or Google Colab.
   - Execute `iris_decision_tree.ipynb`, `mnist_cnn.py`, and `amazon_nlp.py`.

5. **Run Streamlit App**:
   ```bash
   streamlit run streamlit/mnist_streamlit.py
   ```

## Project Details

### Task 1: Classical ML with Scikit-learn
- **Dataset**: Iris Species Dataset
- **Goal**: Train a decision tree classifier to predict iris species.
- **Implementation**:
  - Preprocessed data (checked for missing values, encoded labels).
  - Trained a decision tree using Scikit-learn.
  - Evaluated with accuracy (~0.97), precision (~0.97), and recall (~0.97).
- **Output**: Confusion matrix visualization (`outputs/iris_confusion_matrix.png`).
- **File**: `notebooks/iris_decision_tree.ipynb`

### Task 2: Deep Learning with TensorFlow
- **Dataset**: MNIST Handwritten Digits
- **Goal**: Build a CNN to classify digits with >95% test accuracy.
- **Implementation**:
  - Built a CNN with two convolutional layers, max-pooling, and dense layers.
  - Achieved test accuracy of ~0.98 after 10 epochs.
  - Visualized predictions on 5 test images.
- **Output**: Prediction images (`outputs/prediction_0.png` to `prediction_4.png`), accuracy/loss plots.
- **File**: `notebooks/mnist_cnn.py`

### Task 3: NLP with spaCy
- **Dataset**: Amazon Product Reviews (sample)
- **Goal**: Perform NER and sentiment analysis.
- **Implementation**:
  - Used spaCy’s `en_core_web_sm` model to extract product names and brands.
  - Applied TextBlob for rule-based sentiment analysis (e.g., Positive, Polarity: 0.55).
- **Output**: NER visualization (`outputs/ner_visualization.png`).
- **File**: `notebooks/amazon_nlp.py`

### Task 4: Ethics & Optimization
- **Ethical Considerations**:
  - **MNIST**: Potential bias in generalizing to diverse handwriting styles. Mitigation: Use TensorFlow Fairness Indicators to evaluate across demographic slices.
  - **Amazon Reviews**: Risk of misclassifying nuanced or non-standard English reviews. Mitigation: Enhance spaCy with custom rules for slang/emojis.
- **Troubleshooting**:
  - Debugged a TensorFlow script with input shape mismatches and incorrect loss function.
  - Fixed code ensures proper input preprocessing and uses `categorical_crossentropy`.
- **File**: `notebooks/fixed_tensorflow.py`

### Bonus Task: Streamlit Deployment
- **Goal**: Deploy the MNIST classifier as a web app.
- **Implementation**:
  - Built a Streamlit app to upload and classify handwritten digit images.
  - Preprocesses 28x28 grayscale images and displays predictions.
- **Output**: Screenshot of app interface (`outputs/streamlit_screenshot.png`).
- **File**: `streamlit/mnist_streamlit.py`
- **Live Demo**: 

## Usage
1. **Run Individual Tasks**:
   - Iris: Open `iris_decision_tree.ipynb` in Jupyter/Colab and execute.
   - MNIST: Run `mnist_cnn.py` to train and visualize.
   - Amazon NLP: Run `amazon_nlp.py` to see NER and sentiment results.
   - Debugged Code: Run `fixed_tensorflow.py` to verify fixes.
2. **Launch Streamlit App**:
   ```bash
   streamlit run streamlit/mnist_streamlit.py
   ```
   Upload a 28x28 grayscale digit image to get predictions.

## Ethical Reflections
- **Bias in MNIST**: The dataset may not represent diverse handwriting, potentially affecting performance in global applications. Future work could include augmenting with diverse numeral styles.
- **Bias in Amazon Reviews**: Sentiment analysis may misinterpret cultural nuances. Custom spaCy rules and fairness audits can improve inclusivity.
- **Mitigation Tools**: TensorFlow Fairness Indicators and spaCy’s rule-based systems help ensure equitable model performance.

## Results
- **Iris Classifier**: Achieved 97% accuracy, precision, and recall.
- **MNIST CNN**: Achieved 98% test accuracy, with clear prediction visualizations.
- **Amazon NLP**: Successfully extracted entities (e.g., "Samsung Galaxy") and determined sentiment (e.g., Positive, Polarity: 0.55).
- **Streamlit App**: Functional web interface for digit classification.

## Screenshots
- Iris Confusion Matrix: `outputs/iris_confusion_matrix.png`
- MNIST Predictions: `outputs/prediction_0.png` to `prediction_4.png`
- Amazon NER Visualization: `outputs/ner_visualization.png`
- Streamlit App: `outputs/streamlit_screenshot.png`

## Contributing
- Fork the repository and create a pull request with clear commit messages.

## Acknowledgments
- Datasets: Kaggle, TensorFlow Datasets
- Tools: TensorFlow, PyTorch, Scikit-learn, spaCy, Streamlit
- Team collaboration facilitated by GitHub and LMS Community.

