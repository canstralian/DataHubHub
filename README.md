# ML Dataset & Code Generation Manager

A comprehensive platform for ML dataset management and code generation with Hugging Face integration.

## Features

- **Dataset Management**: Upload, explore, and manage machine learning datasets
- **Data Visualization**: Visualize dataset statistics and distributions
- **Code Generation**: Fine-tune models for code generation tasks
- **Code Quality Tools**: Improve code quality with integrated formatters, linters, and type checkers

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: SQLite (via SQLAlchemy)
- **ML Integration**: Hugging Face Transformers, Datasets
- **Visualization**: Plotly, Matplotlib

## Project Structure

```
.
├── app.py                     # Main application entry point
├── components/                # UI components
│   ├── code_quality.py        # Code quality tools
│   ├── dataset_preview.py     # Dataset preview component
│   ├── dataset_statistics.py  # Dataset statistics component
│   ├── dataset_uploader.py    # Dataset upload component
│   ├── dataset_validation.py  # Dataset validation component
│   ├── dataset_visualization.py # Dataset visualization component
│   └── fine_tuning/           # Fine-tuning components
│       ├── finetune_ui.py     # Fine-tuning UI
│       └── model_interface.py # Model interface
├── database/                  # Database configuration
│   ├── models.py              # Database models
│   └── operations.py          # Database operations
├── utils/                     # Utility functions
│   ├── dataset_utils.py       # Dataset utilities
│   ├── huggingface_integration.py # Hugging Face integration
│   └── smolagents_integration.py # SmolaAgents integration
└── assets/                    # Static assets
```

## Deployment

This application is designed to be deployed as a Hugging Face Space.

### Hugging Face Space Deployment

1. Fork this repository
2. Create a new Hugging Face Space
3. Connect the forked repository to your Space
4. The application will be deployed automatically

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install streamlit pandas numpy plotly matplotlib scikit-learn SQLAlchemy huggingface-hub datasets transformers torch
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Configuration

- `.streamlit/config.toml`: Streamlit configuration
- `.streamlit/secrets.toml`: Secrets and API keys
- `huggingface-spacefile`: Hugging Face Space configuration

## API Keys

To use the Hugging Face integration features, add your Hugging Face API token to `.streamlit/secrets.toml`:

```toml
[huggingface]
hf_token = "YOUR_HF_TOKEN"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.