# RᴀLLᴇ

RᴀLLᴇ is an accessible framework for developing and evaluating retrieval-augmented large language models (R-LLMs).  
An overview of the main uses of RᴀLLᴇ on GUI is presented in [this video](https://youtu.be/JYbm75qnfTg).


It is developed at [the Institute of Memory Technology Research and Development](https://www.kioxia.com/en-jp/rd/organization/memory-technology-rd.html) of [Kioxia Corporation](https://www.kioxia.com/en-jp/top.html).

## Key Features

- *Easy development and testing*: users can easily select, combine, and test various retrievers and LLMs, including open-source models, within a graphical interface.
- *Objective evaluation of R-LLMs*: RᴀLLᴇ provides reproducible experiments with objective benchmarks/metrics, enabling objective assessments of R-LLM performance.
- *Transparent prompt engineering*: all input (prompts) and output of an LLM are visible to the user, allowing for easy exploration and optimization of prompts.

## Usage

Getting Started:

- Installation instruction: [INSTALL.md](INSTALL.md).
- Document indexing: [here](docs/indexing.md).

Using RᴀLLᴇ:

- Guides on GUI: [here](docs/gui_usage.md).
- Evaluation with a Python script: [here](docs/evaluation.md).
- Review the evaluation results with MLflow: [here](docs/mlflow.md)

Note: evaluation experiments can be performed both [through the GUI](docs/gui_usage.md#evaluate-and-save-config) and using the script.

<!-- ## Star History -->

<!-- [star-history.com](https://star-history.com/) -->

## Reference

Reference to cite when you use RᴀLLᴇ in a research paper:

```bibtex
@misc{ralle,
      title={RaLLe: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models}, 
      author={Yasuto Hoshi and Daisuke Miyashita and Youyang Ng and Kento Tatsuno and Yasuhiro Morioka and Osamu Torii and Jun Deguchi},
      url={https://arxiv.org/abs/2308.10633},
      year={2023},
      eprint={2308.10633},
      publisher={arXiv}
}
```

## License

RᴀLLᴇ is MIT-licensed, refer to the [LICENSE file](../LICENSE) for more details.

