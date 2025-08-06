# AI Risk Assistant: LLM Fine-Tuning with Databricks

<p align="center">
  <a href="https://www.databricks.com/" target="_blank">
    <img src="https://img.shields.io/badge/Data_Platform-Databricks-orange.svg?logo=databricks" alt="Databricks as Data Platform">
  </a>
  <a href="https://docs.databricks.com/en/dev-tools/bundles/index.html" target="_blank">
    <img src="https://img.shields.io/badge/Dev_Framework-Databricks_Asset_Bundles-blue.svg?logo=databricks" alt="Databricks Asset Bundle as Development Framework">
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.12.x-3776AB.svg?logo=python" alt="Python 3.12.x">
  </a>
  <a href="https://docs.databricks.com/en/notebooks/index.html" target="_blank">
    <img src="https://img.shields.io/badge/Interface-Notebook-F37626.svg?logo=jupyter" alt="Notebook">
  </a>
</p>

<p align="center">
  <b>Fine-tuning Capabilities</b><br>
  <a href="https://deepmind.google/technologies/gemini/" target="_blank">
    <img src="https://img.shields.io/badge/Commercial_LLM-Gemini_2.5-4285F4.svg?logo=google-gemini" alt="Finetune Gemini 2.5">
  </a>
  <a href="https://ai.meta.com/blog/meta-llama-3/" target="_blank">
    <img src="https://img.shields.io/badge/Open_Source_LLM-Llama_3-brightgreen.svg?logo=meta" alt="Finetune Llama 3">
  </a>
  <a href="https://qwen.vl-gate.com/blog/qwen2" target="_blank">
    <img src="https://img.shields.io/badge/Open_Source_LLM-Qwen_3-FF6A00.svg?logo=alibabacloud" alt="Finetune Qwen 3">
  </a>
</p>

This repository contains a robust framework for fine-tuning Large Language Models (LLMs) for specialized tasks, such as risk analysis and mitigation. The entire MLOps process is orchestrated using **Databricks Asset Bundles (DAB)**, providing a reproducible, production-ready workflow on the Databricks Data Intelligence Platform.

This project supports fine-tuning for both leading commercial and open-source models, including Google's **Gemini 2.5** and Meta's **Llama 3** & Alibaba's **Qwen 3**.

## Fine-tuning Data Pipeline

The foundation of our approach is a curated set of Python libraries, each serving a specific and critical function in the LLM fine-tuning pipeline.

  * **`transformers`**: Provided by Hugging Face, this is the cornerstone library for interacting with pre-trained models. It supplies the interfaces for loading models (e.g., Llama 3, Qwen 3) and their corresponding tokenizers, which are essential for converting text into a format the model can process.
  * **`peft`** (Parameter-Efficient Fine-Tuning): Another Hugging Face library, `peft` is the core engine for our efficient fine-tuning strategy. It implements various PEFT methods, most notably Low-Rank Adaptation (LoRA), which allows us to fine-tune a model by training only a minuscule fraction of its parameters, drastically reducing computational requirements.
  * **`bitsandbytes`**: This library is crucial for quantization, the process of reducing the precision of a model's weights. We use it to implement 4-bit quantization for QLoRA, which further slashes the memory footprint of the base model, making it possible to load billion-parameter models into a single GPU's VRAM.
  * **`trl`** (Transformer Reinforcement Learning): From Hugging Face, `trl` provides high-level abstractions for training, including the `SFTTrainer` (Supervised Fine-tuning Trainer). This class simplifies the training loop by handling data collation, optimization, and logging, allowing us to focus on the model and data rather than boilerplate code.
  * **`accelerate`**: This library provides a seamless and unified API for distributed training across multiple GPUs or TPUs. While our primary implementation targets a single GPU, `accelerate` works in the background to manage device placement and optimize performance.
  * **`unsloth`**: This is a key strategic component of our architecture. `unsloth` is a specialized library that dramatically accelerates the fine-tuning process (up to 2x faster) and reduces memory consumption (by up to 70%) through hand-written Triton kernels and a manual backpropagation engine. The inclusion of `unsloth` is a strategic enabler that transforms this project into a practical workflow that can be executed on widely available hardware.
  * **`datasets`**: The standard Hugging Face library for loading, manipulating, and streaming large datasets efficiently. We use it to manage our instruction-formatted data.
  * **`kaggle`**: The official Kaggle API, which allows for the programmatic download and management of datasets directly within our Python environment, automating the data acquisition process.

## Pipeline Architecture with Databricks Asset Bundles

The implementation is developed with **Databricks Asset Bundles** to create fine-tuning pipelines and process through the steps below to run fine-tuning tasks on LLMs.

### Standard Pipeline Tasks

The bundle defines a series of tasks that form a complete data preparation and model training workflow:

1.  **`Pull_dataset_from_kaggle`**: Retrieves the raw dataset from the Kaggle Platform using the Kaggle API.
2.  **`Generate_instruction_dataset`**: Applies a set of heuristic rules to generate a "gold standard" response for each data sample and formats the entire row into an `instruction, context, response` structure suitable for supervised fine-tuning.
3.  **`Unify_finetune_dataset`**: Processes and unifies disparate data sources into a single, consistently structured dataset ready for training.
4.  **`Qlora_configuration`**: Configures the QLoRA (Quantized Low-Rank Adaptation) parameters using the `peft` library for memory-efficient training.
5.  **`Tokenizer`**: Initializes and configures the appropriate tokenizer for the target model, which breaks down text into smaller, known sub-word units.
6.  **`Finetune_model`**: Executes the core fine-tuning job. This task is specialized for each model:
      * `finetune_gemini`
      * `finetune_llama`
      * `finetune_qwen`

### Example Workflow Visualizations

The tasks defined in the bundle can be chained together to create end-to-end workflows for specific use cases, as illustrated in the project documentation.

  * **Figure 4.1: Pipeline Fine-tune for Risk Analysis**
  * **Figure 4.2: Pipeline Fine-tune for Risk Mitigation**

## Getting Started

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/tankhuu/ai-risk-assistant-finetune.git
    cd ai-risk-assistant-finetune
    ```

2.  **Configure Databricks CLI**
    Ensure your Databricks CLI is installed and configured to connect to your workspace.

3.  **Deploy the Bundle**
    Use the Databricks CLI to validate and deploy the bundle. This will provision the necessary jobs and workflows in your Databricks workspace.

    ```bash
    # Validate the bundle configuration
    databricks bundle validate

    # Deploy the bundle to Databricks
    databricks bundle deploy
    ```

4.  **Run the Pipeline**
    Navigate to the **Workflows** tab in your Databricks workspace to trigger and monitor the deployed fine-tuning pipeline.

## Source Code

For more details on the implementation, please visit the full source code repository:
[https://github.com/tankhuu/ai-risk-assistant-finetune](https://github.com/tankhuu/ai-risk-assistant-finetune)