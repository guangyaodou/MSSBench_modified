# CIS 6200 Final Project - (Multimodal Situational Safety) 

Our project is based on the MSSBench dataset, which is a benchmark for evaluating the safety of multimodal large language models (MLLMs) in various tasks. The dataset includes two main tasks: Chat Task and Embodied Task.
More details can be found [here](https://github.com/eric-ai-lab/MSSBench/tree/main).

## Dataset Structure
The [Dataset](https://huggingface.co/datasets/kzhou35/mssbench/tree/main) can be downloaded from Hugging Face.


## Evaluation
You can evaluate different MLLMs by running our evaluation code [inference.py](inference.py) and changing the "--mllm" parameter: 

```sh
python inference.py --mllm gemini --data_root xxx --output_dir xxx
```

The deployment of the model can refer to [models](models). For proprietary models, please set up your API key first.

For OpenAI models, make sure that you have the OpenAI API key set up in your environment. You can do this by running:

```sh
export OPENAI_API_KEY="<your_openai_api_key>"
```

or add it to your `.bashrc` file.
