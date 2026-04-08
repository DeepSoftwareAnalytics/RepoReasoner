# **RepoReasoner: Evaluating Repository-Level Reasoning Ability of Large Language Models**

## 📝 Overview
> Current benchmarks for code reasoning primarily focus on function-level, often overlooking real-world software codebase granularity. This leaves a gap in understanding the ability of Large Language Models (LLMs) in code reasoning.
To provide a more comprehensive and real-world software development evaluation scenario, we construct **RepoReasoner**.

This repository contains the complete code for constructing RepoReasoner. The pipeline is designed to generate instances from open-source Python repositories for two primary tasks: **Output Prediction** and **Call Chain Prediction**. The entire process is automated, from repository selection and filtering to data generation, rewriting, and final evaluation using LLMs.

## 🌈 Features

- 🤖 **Automated Benchmark Pipeline**: An automated workflow about repository and data filtering to benchmark's instances generation.
- 🎯 **Repository-Level Task Evaluation**: Generates instances for two repository-level reasoning tasks: **Output Prediction** and **Call Chain Prediction**, focusing on micro-level and macro-level software development scenarios.
- ✍️ **Semantic Data Rewriting**: Augments the dataset by generating semantically equivalent but syntactically different code to test model robustness.
- ⚙️ **Flexible LLM Integration**: Supports evaluation using both commercial API-based models and local Hugging Face models.


## **Prerequisites**

Before you begin, ensure you have the following installed:
*   Python 3.8+
*   Docker
*   An OpenAI-compatible API key or a local Hugging Face model.

## 🔧**Setup**

1.  **Install Python dependencies:**

    *(Note: The `requirements.txt` file is used to construct environment in docker It serves LLMs' interaction (a lightweight, LLM-powered tool which is used in docker) in auto env-setup & auto I/O ReWriting. To avoid confusion, we name RepoReasoner's dependencies file `project.txt`)*

    You need to use `project.txt` like 
    ```bash
    pip install -r project.txt
    ```

2.  **Configure API Access:**
    If using an API-based model, create a file named `API_KEY.txt` in the `Exec-Based Filtering` & `Inference_by_LLMs`, and place your key inside it.

3.  **Python Repositories:**
    Place the target Python repositories inside the `python_repos/` directory. And a file `python_repos.txt` filled with there each repository name a line.

    **NOTE: In order to make the program logic easy to implement and more efficient, we have added a lot of hard coding to the code for folder naming, so we hope you don't to use custom folder names.**


## 📊**Evaluation**
All result of evaluation are organized within the `evaluation_result/`directory. 

You could use `python evaluation_output_prediction.py` & `python evaluation_callchain_prediction.py` to run the evaluation script.

If want to evaluation on your results, you could put scripts into your experiments' directory and run them.



## 📚**Benchmark Pipeline: Step-by-Step**

This section details the sequential execution of the scripts to generate the final benchmark dataset.


### **Exec-Based Filtering**

We put all code about 'Stage II: Execution-Based Filtering' in `Exec-Based Filtering/`.

This crucial step uses a containerized environment to validate each repository and collect dynamic runtime information, such as call chains.

You need to change the base_url in `runnable_agnent_batch/generator.py`. We upload the code used for tool and API configuration when constructing Docker images.

The auto env-setuper & auto I/O-Rewriter need a `API_KEY.txt`.

**Script:**
*   `docker_runner.py`


### **Raw Data Collection**

We put all code about data collection (include 'Stage IV: Instance Collection') in `Instance Collection/`.

This step parses the original source code of the test files to extract potential benchmark instances. 

**Script:**
*   `data_collection.py`


### **Data Rewriting**

We put all code about 'Stage III: Data Rewriting' in `Data Rewriting/`

To augment the dataset and test model robustness, we generate a parallel corpus of rewritten test cases that are semantically equivalent but syntactically different.

**Scripts:**
*   `rewrite_runner.py`
*   `rewrite_data_collection.py`
*   `rewrite_data_align.py`

### **Instance Collection**

The statically extracted `ground_truth` may not always represent the precise runtime value. This step executes the code to capture the exact runtime value.

**Scripts:**
*   `groundtruth_collection.py`: Injects code to capture and serialize the runtime value of the ground truth expression.
*   `groundtruth_supplement.py`: Processes the collected runtime values and adds them as valid answers to the dataset.

### **Running Experiments with LLMs**

With the final dataset prepared, you can now run the evaluation experiments. We provide scripts for both API-based and locally-hosted models.

Prerequisite - BM25 Retrieval: Before running output prediction, you must first index the repositories to enable retrieval-augmented context.
***Script:**
*   `related_files_collection.py`

**Experiment Scripts:**
*   **Output Prediction:**
    *   `output_prediction.py` (API-based models)
    *   `output_prediction_model_inference.py` (Local Hugging Face models)
    *   Task: Given the masked code and relevant context files, the model must predict the correct value for the masked assertion.

*   **Call Chain Prediction:**
    *   `callchain_prediction.py` (API-based models)
    *   `callchain_prediction_model_inference.py` (Local Hugging Face models)
    *   Task: Given a test file, the model must predict the list of other source files that are executed as part of its call chain.

The results of all experiments (predictions, interaction logs) are saved to the `experiments_output_*` and `experiments_callchain_*` directories.