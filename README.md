# Textbook is all you need
This repository is a reproduction of the experiment from the paper ["Textbooks are all you need"](https://arxiv.org/pdf/2306.11644). 

The goal of the project is to fine-tune a Transformer-based model on synthetic data to improve its performance on code completion tasks in Kotlin language.

## Authors
- [Nemanja Vujadinovic](https://github.com/vujadinovicn)

## Modification from the paper
The original paper trained and fine-tuned the `phi-1` model. In this project, we utilize `deepseek-ai/deepseek-coder-1.3b-base` model. Additionally, we fine-tune our model on synthetic data of Kotlin code and evaluate it on `JetBrains/Kotlin_HumanEval` dataset.

## Implementation
Here's how we reproduced the original paper:

### Evaluation of the pretrained model
To get a sense of how well the pretrained `deepseek-coder` model performs out of the box, we ran it on the HumanEval for Kotlin benchmark. Code for evaluation is available in `evaluation.ipynb` and it's a slightly modified version of the code provided in [Hugging Face documentation](https://huggingface.co/datasets/JetBrains/Kotlin_HumanEval). Model got a score of `0.28`.

### Code translation
Next, we had to generate synthetic Kotlin data. To do this, we chose the jinaai/code_exercises dataset, which contains code written in Python. Since we required Kotlin code, we had to translate the Python code to Kotlin. We explored two approaches:
- Translate the code using LLMs (in Python): We initially tried using the `CodeT5` transformer to translate the code to Kotlin. However, this approach was unsuccessful, as the outputs were not nearly accurate enough for our needs. Code is available in `python2kotlin_translation` notebook.
- Obtain the Kotlin code by prompting ChatGPT: We have manually constructed prompts by which we have instructed ChatGPT to translate Python code into Kotlin. This approach was successful, so we proceeded with it. Given that each translation had to be done manually, we limited the dataset to 100 examples. Collected data is available at `data/python_to_kotlin_data.jsonl'.


### Model fine-tuning
Now, we had to fine-tune our `deepseek-coder` model to improve its previous benchmark score. Our dataset included two columns - `"problem"` and `"solution"`. We tokenized both sequences, setting the tokenized `"problem"` as the model input and the tokenized `"solution"` as the label. Code for fine-tuning the model is available in `fine-tuning.ipynb`.
Unfortunately, due to computational limitations, we were unable to complete the fine-tuning process. This part of the project will be finished in the near future.

### Re-evaluation of the fine-tuned model
Since we haven’t been able to fine-tune the model yet, we don't have updated results. This task is planned for completion soon.

## Results 
The results are shown in the table below.
| | `deepseek-coder` | fine-tuned `deepseek-coder`|
|:-----------------:|:-----------------:|:-------------------:|
| HumanEval score|0.28        | N/A           |


## How to run
The project requires Python 3.8+ and the dependencies listed in requirements.txt. 
Install them using:

```bash
pip install -r requirements.txt
```
Additionally, you have to download `Kotlin` and `Java JDK` and add them to your system PATH. For `mxeval`, run the following commands:
```
git clone https://github.com/amazon-science/mxeval.git
pip install -e mxeval
```
Link to full documentation on `mxeval` is available [here](https://github.com/amazon-science/mxeval).

### Usage
- Evaluate pretrained model using `evaluate.ipynb` notebook.
- Try code translation using LLMs using `python2kotlin_translation` notebook. If you want to use our collected data, check out `data/python_to_kotlin_data.jsonl'.
- Fine tune the model using `fine_tuning.ipynb` notebook

## Future improvements
- Complete the fine-tuning phase of the project.
- Re-evaluate the fine-tuned model on the same benchmark.
- Create a report and analysis of the results.
- Reproduce the paper using other models.
