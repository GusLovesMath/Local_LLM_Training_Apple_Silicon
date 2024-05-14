# Local LLM Training on Apple Silicon - Project README

This repository contains the resources and documentation for the project "Local LLM Training on Apple Silicon", where the Llama3 model was fine-tuned to efficiently solve verbose mathematical word problems on an Apple Silicon device with 16 GPUs. The project demonstrates the application of the MLX library and Metal API to achieve high computational performance and privacy on non-traditional hardware platforms.

## Repository Contents
- **LLM_Local_Training_Llama3.ipynb**: Jupyter notebook containing all the code for setting up, training, and evaluating the LlaMATH3 model.
- **AIChatbotWithLLM_SLIDES.pdf**: Presentation slides detailing the project's approach, architecture, and outcomes.
- **AIChatbotWithLLM_Report.pdf**: Comprehensive report discussing the project in detail.
- **AIChatbotWithLLM_Report_Summary_onePage.pdf**: One-page summary of the project report for quick reference.
- **app.py**: Updated GUI application file to replace the original in the [chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx.git) repository for enhanced user interaction.
- **LlaMATH-3-8B-Instruct-4bit.yaml**: Configuration file to be added to the `../chat-with-mlx/chat_with_mlx/models/config` directory for using the custom trained model.

## Installation
To set up the project environment and run the models, you will need to install the following software and libraries:

```bash
conda create -n localLLM python=3.11
activate localLLM
pip install mlx-llm
pip install torch==2.3.0
pip install transformers==4.40.1
pip install datasets==2.19.0
pip install pandas==2.2.2
```

## Usage
To use the trained LlaMATH3 model for generating responses to mathematical prompts, follow these steps:

```python
from mlx_lm import load, generate

# Load the model
model, tokenizer = load("GusLovesMath/LlaMATH-3-8B-Instruct-4bit")

# Example prompt
prompt = """
Q: A new program had 60 downloads in the first month.
The number of downloads in the second month was three times as many as the first month,
but then reduced by 30% in the third month. How many downloads did the program have total over the three months?
"""

# Generate response
response = generate(model, tokenizer, prompt=prompt, max_tokens=132, temp=0.0, verbose=True)
print('LlaMATH Response:', response)
```

## Model Details
- **Source**: The model was converted to MLX format from `mlx-community/Meta-Llama-3-8B-Instruct-4bit` using mlx-lm version 0.12.1.
- **Training Hardware**: Apple M2 Pro chip with 16GB of RAM, 16 GPUs, and CPUs.
- **Model Card**: For more detailed information about the model's capabilities and training, refer to the original [model card](https://huggingface.co/GusLovesMath/LlaMATH-3-8B-Instruct-4bit).

## Interface with [chat-with-mlx](https://github.com/qnguyen3/chat-with-mlx.git) and updated `app.py` File
<img width="900" alt="Screenshot 2024-05-14 at 2 11 21 PM" src="https://github.com/GusLovesMath/Local_LLM_Training_Apple_Silicon/assets/109978367/2f310164-f38e-429e-9476-7b471931d652">
