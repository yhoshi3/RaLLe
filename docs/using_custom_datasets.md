# Using Custom Datasets

In addition to utilizing KILT datasets, RᴀLLᴇ enables you to develop and evaluate retrieval-augmented LLMs on your own datasets.
This documentation outlines the pre-processing steps required to prepare a custom corpus and a custom QA dataset for use with RᴀLLᴇ.


- [Using a custom knowledge source](#using-a-custom-knowledge-source)
  - [Preprocessing](#preprocessing)
  - [Indexing](#indexing)
    - [Dense retrieval](#dense-retrieval)
    - [BM25](#bm25)
  - [Update the config file to use the custom corpus](#update-the-config-file-to-use-the-custom-corpus)
- [Using a custom QA dataset](#using-a-custom-qa-dataset)
- [Development on GUI](#development-on-gui)
- [Evaluation](#evaluation)

If your QA dataset utilizes the KILT Wikipedia as its knowledge source, you can use the QA dataset by simply specifying its name and path in the [configuration file](../scripts/configs/base_settings/datasets.json), following proper preprocessing for the QA dataset as described below.

## Using a custom knowledge source

### Preprocessing

Your custom corpus should be a .tsv file in the format shown in the following example.

```tsv
id	text	title
1	Alice likes adventures.	Alice
2	Bob likes reading books.	Bob
3	Charlie is a free-spirited artist.	Charlie

```

Note:

- The header ('id text title') is removed during encoding (that is, this line will not be included in the search results).
- A unique document ID (a contiguous integer) should be assigned to each line.

Here we provide a brief example for constructing a custom corpus.

First, create a new file named `my_corpus.tsv` in `${my_corpus_dir}`.

```bash
export data=/path/to/data
export my_corpus_dir=${data}/text/corpus/my_corpus
mkdir -p ${my_corpus_dir}
```

Here is an example of creating a corpus consisting of the three texts shown above.

```python
texts = [
    "Alice likes adventures.",
    "Bob likes reading books.",
    "Charlie is a free-spirited artist."]

titles = ["Alice", "Bob", "Charlie"] # or [""]*3 if unnecessary

corpus_path = 'data/text/corpus/custom/my_corpus.tsv'
with open(corpus_path, 'w') as file_out:
    file_out.write('id{}text{}title{}'.format('\t', '\t', '\n')) # header
    for i, (text, title) in enumerate(zip(texts, titles)):
        file_out.write(f'{i+1}\t"{text}"\t{title}\n')
```

Next, create a file to map from document ID to corpus ID.
This file is required for using KILT scripts without modifying them.

```python
import pickle
n_documents = 3  # total number of texts
output = {i:i for i in range(1, n_documents + 1)}

output_path = 'data/text/corpus/custom/doc_id_to_wikipedia_id_mapping.p'

with open(output_path, 'wb') as f:
    pickle.dump(output, f)
```

### Indexing

Here we demonstrate indexing procedures for both dense (e5 model) and sparse (BM25) retrieval.
The title of a text is prepended with a comma, '{title}, {text}', which is then indexed.

#### Dense retrieval

First, encode the custom corpus with [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) as an embedding model (see also: [indexing.md](indexing.md)).
Running the script below will generate a NumPy memmap file of the embeddings and a tsv file without a header (`my_corpus_modified.tsv`).


```bash
export emb_dir=${data}/dense/e5/8b/corpus/my_corpus
CUDA_VISIBLE_DEVICES=0 python scripts/make_embeddings.py \
    --data_dir ${my_corpus_dir} \
    --corpus_name my_corpus \
    --out_dir ${emb_dir} \
    --use_8bit
```

`--corpus_name` : the name of corpus tsv file without ".tsv"

Next, build a Faiss flat index from the embedding.

```bash
export n_docs=3 # total number of texts
python scripts/build_faiss_index.py \
    --embeddings ${emb_dir}/full_1024_${n_docs}.npmmap \
    --index_type flat
```

The built index will be stored in `data/dense/e5/8b/corpus/custom/index/faiss-flat/full_1024_3.index`

#### BM25

Run the following command for indexing (see also: [indexing.md](indexing.md#build-bm25-index-optional-2-minutes)).

```bash
python scripts/build_bm25_index.py --wiki_passages_file ${my_corpus_dir}/my_corpus_modified.tsv
```

This script create the following directory:

`${data}/bm25/corpus/my_corpus/my_corpus_modified`

Note that, the directory name is automatically created from `${my_corpus_dir}` by replacing '/text/' with '/bm25/'.

### Update the config file to use the custom corpus

To enable retrieving from the custom corpus,

1. Add "my_corpus" to [scripts/configs/base_settings/corpora.json](../scripts/configs/base_settings/corpora.json)

```json
{
    ...,
    "my_corpus": {
        "corpus_path": "data/text/corpus/my_corpus/my_corpus_modified.tsv",
        "delimiter": "\t",
        "col_doc_id": 0,
        "col_text": 1,
        "col_title": 2,
        "doc_id_to_wikipedia_id_mapping_path": "data/text/corpus/my_corpus/doc_id_to_wikipedia_id_mapping.p"
    },
}
```

Note: Each line of the tsv file is structured as [doc_id, text, title], with `col_doc_id` corresponding to column 0, `col_text` to column 1, and `col_title` to column 2.

2. Add the built indices to [scripts/configs/base_settings/retrievers.json](../scripts/configs/base_settings/retrievers.json)

```json
    ...,
    "e5-8b_flat_custom": {
        "type": "faiss",
        "query_encoder_name": "e5-8b",
        "index_path": "data/dense/e5/8b/corpus/my_corpus/index/faiss-flat/full_1024_3.index",
        "corpus_name": "my_corpus"
    },
    "bm25_custom": {
        "type": "bm25",
        "index_path": "data/bm25/corpus/my_corpus/my_corpus_modified",
        "corpus_name": "my_corpus"
    },
```

## Using a custom QA dataset

You can utilize your QA dataset with RaLLe by preparing it in JSONL format with the identical structure as the KILT datasets.

Example:

```jsonl
{"id": "1", "input": "Who is Charlie?", "output": [{"answer": "artist", "provenance": [{"wikipedia_id": "3", "title": "Charlie"}]}]}
{"id": "2", "input": "What does Alice like?", "output": [{"answer": "adventure", "provenance": [{"wikipedia_id": "1", "title": "Alice"}]}]}

```

The `wikipedia_id` serves as the document ID within the custom corpus, if the Wikipedia ID and document ID are identical, as in `my_corpus`.

Add the development set and test set to [scripts/configs/base_settings/datasets.json](../scripts/configs/base_settings/datasets.json)

```json
{
    "develop": {
        ...,
        "my_qa_data": "/path/to/my_qa_dev.jsonl"
    },
    "test": {
        ...,
        "my_qa_data": "/path/to/my_qa_test.jsonl"
    }
}
```

Notes:

- Use a consistent name for both the development and test sets (e.g., "my_qa_data").
- Include the development set to perform development on the GUI, including prompt engineering.

## Development on GUI

Once you have finished adding the necessary settings to the configuration file, you can proceed to create an inference chain and design the appropriate prompts within the GUI using your tailored dataset.
For additional assistance, please refer to our documentation available in [gui_usage.md](gui_usage.md).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/chat.py
```

On the launched GUI,

- You can load the index that you created under the `Retrievers` dropdown list in the `Load Models` tab.
- You can choose your QA dataset from the `Dataset` dropdown list in the `Develop Chain` tab.
- Once you have developed a promising chain with effective prompts, you can save the configuration of the chain by clicking the `Save Config` button in the `Config` tab.

## Evaluation

To configure your inference chain for evaluation, place a configuration file similar to the one below in `scripts/configs/experiment_settings/my_qa`.
If you have an existing configuration file from previous chain development, you can use it.

```json
{
    "chain_config": {
        "dataset": {
            "dataset_name": "my_qa_data",
            "num_evaluate": -1,
            "batch_size": 20
        },
        "len_chain": 2,
        "chain": [
            {
                "prompt_template": "{question}",
                "function": "Retriever",
                "retriever_name": "e5-8b_flat_custom",
                "npassage": 1,
                "f-strings_or_eval": "f-strings"
            },
            {
                "prompt_template": "Referring to the following document, answer \"{question}?\" in 5 words or less.\n\n{response[0]}\n\nAnswer:",
                "function": "LLM",
                "llm_name": "llama-2-13b-chat",
                "f-strings_or_eval": "f-strings"
            }
        ]
    }
}
```

- `dataset_name` : the name of a custom QA dataset defined in [scripts/configs/base_settings/datasets.json](../scripts/configs/base_settings/datasets.json)
- `retriever_name` : a name of retriever defined in [scripts/configs/base_settings/retrievers.json](../scripts/configs/base_settings/retrievers.json). In the above example, we use dense retrieval with the custom index.
- `prompt_template` : the most promising prompt templates developed on the dev set of your QA dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/experiment.py \
    --config_exp_path=scripts/configs/experiment_settings/my_qa
```

The evaluation result can be viewed either in `outputs/results.json` or on the MLflow GUI using the following command:

```bash
mlflow ui
```
