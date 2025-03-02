# ML Dataset & Code Generation Manager

A comprehensive platform for ML dataset management with Hugging Face integration, fine-tuning capabilities, and code quality tools.

## Features

- **Dataset Management**: Upload, analyze, and validate ML datasets
- **Hugging Face Integration**: Search and load datasets from Hugging Face Hub
- **Dataset Visualization**: Interactive charts and statistics for data exploration
- **Fine-Tuning**: Custom fine-tuning of code generation models
- **Database Integration**: Persistent storage for datasets and training jobs
- **Code Quality Tools**: Tools for code linting, formatting, and testing

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas, NumPy, Plotly
- HuggingFace libraries (transformers, datasets)
- SQLAlchemy

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/ml-dataset-code-generation-manager.git
cd ml-dataset-code-generation-manager
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:5000`.

## Project Structure

- `main.py`: Main application entry point
- `assets/`: CSS and static assets
- `components/`: UI components and modules
  - `dataset_uploader.py`: Dataset upload functionality
  - `dataset_preview.py`: Dataset preview component
  - `dataset_statistics.py`: Statistical analysis of datasets
  - `dataset_validation.py`: Dataset validation tools
  - `dataset_visualization.py`: Data visualization components
  - `fine_tuning/`: Fine-tuning components and models
  - `code_quality.py`: Code quality tools
- `database/`: Database models and operations
- `utils/`: Utility functions and helpers

## Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face Hub
2. Select the Streamlit SDK
3. Connect your GitHub repository or upload the files directly
4. The Space will automatically deploy the application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for their amazing models and datasets
- [Streamlit](https://streamlit.io/) for the awesome UI framework