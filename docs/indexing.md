# Indexing for quick start

In this document, we show how to prepare document index using a small subset of KILT passages for quick start. You can make full size index of KILT passages or your own documents in the same way.

## prepare data directory

```
export data=/path/to/data
mkdir -p ${data}
```

## download data

### KILT corpus (knowledge source)

```
export kilt_corpus_dir=${data}/text/corpus/kilt
mkdir -p ${kilt_data_dir}
```

Download the following files in the `kilt_corpus_dir` directory from KILT web page. See [KILT/retrievers](https://github.com/facebookresearch/KILT/tree/main/kilt/retrievers) for the detail.

- kilt_w100_title.tsv (direct link: http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv)
- mapping_KILT_title.p (direct link: http://dl.fbaipublicfiles.com/KILT/mapping_KILT_title.p)

### KILT dataset

```
cd KILT
mkdir data
python scripts/download_all_kilt_data.py
python scripts/get_triviaqa_input.py
cd ..
```

See [KILT](https://github.com/facebookresearch/KILT) for the detail.

## make doc_id to wikipedia_id mapping file

```
python scripts/doc_id_to_wikipedia_id.py --corpus_path ${kilt_corpus_dir}/kilt_w100_title.tsv --mapping_path ${kilt_corpus_dir}/mapping_KILT_title.p --output_path ${kilt_corpus_dir}/kilt/doc_id_to_wikipedia_id_mapping.p
```

## preprocess for corpus and make embeddings

In this example, we use [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) as an embedding model. We process only 499992 passages out of 22 million passages. This takes about 25 mins with single A100(40GB).

```
export emb_dir=${data}/dense/e5/8b/corpus/kilt
CUDA_VISIBLE_DEVICES=0 python scripts/make_embeddings.py --data_dir ${kilt_corpus_dir} --subset 499992 --out_dir ${emb_dir} --use_8bit
```

This script creates the following files.

- `${kilt_corpus_dir}/kilt_w100_title_modified_subset_499992.tsv`
    - Header line and unnecessary quatations are removed from original tsv file.
- `${emb_dir}/subset_1024_499992.npmmap`


## build faiss index

- flat index (~1 second)
    - ```
      python scripts/build_faiss_index.py --embeddings ${emb_dir}/subset_1024_499992.npmmap --index_type flat
      ```
    - This script creates the following index file.
        - `${emb_dir}/index/faiss-flat/subset_1024_499992.index`

- hnsw index (~30 seconds)
    - ```
      python scripts/build_faiss_index.py --embeddings ${emb_dir}/subset_1024_499992.npmmap --index_type hnsw
      ```
    - This script creates the following index file.
        - `${emb_dir}/index/faiss-hnsw/subset_m16_efc40_1024_499992.index`

## build BM25 index (optional, ~2 minutes)

```
python scripts/build_bm25_index.py --wiki_passages_file ${kilt_corpus_dir}/kilt_w100_title_modified_subset_499992.tsv
```

This script create the following index directory.

- `${data}/bm25/corpus/kilt/kilt_w100_title_modified_subset_499992`
    - Note that the directory name is automatically created from `${kilt_corpus_dir}` by replacing '/text/' with '/bm25/'.

## build DiskANN index (optional, ~20 minutes)

```
python scripts/build_diskann_index.py --npmmap_file ${emb_dir}/subset_1024_499992.npmmap --index_dir ${emb_dir}/index/diskann/subset_499992 --num_threads 16
```

## Pickup answerable questions from KILT dataset

When we use the build indice for the small subset of corpus, we can not find useful information to answer the most of questions in datasets. So here, we pick up only answerable questions from dataset.

```
python scripts/question_pickup.py --dataset_path KILT/data/nq-train-kilt.jsonl --corpus_path ${kilt_corpus_dir}/kilt_w100_title_modified_subset_499992.tsv --mapping_path ${kilt_corpus_dir}/mapping_KILT_title.p --number_of_questions 100 --output_path data/text/query/kilt/nq-train-kilt_100.jsonl

python scripts/question_pickup.py --dataset_path KILT/data/nq-dev-kilt.jsonl --corpus_path ${kilt_corpus_dir}/kilt_w100_title
_modified_subset_499992.tsv --mapping_path ${kilt_corpus_dir}/mapping_KILT_title.p --n
umber_of_questions 100 --output_path data/text/query/kilt/nq-dev-kilt_100.jsonl
```

## data directories

Below is the directory structure after completing all the above indexing processes .

```
${data}
├── bm25
│   └── corpus
│       └── kilt
│           └── kilt_w100_title_modified_subset_499992
│               └── (BM25 index files)
├── dense
│   └── e5
│       └── 8b
│           └── corpus
│               └── kilt
│                   ├── index
│                   │   ├── diskann
│                   │   │   └── subset_499992
│                   │   │       └── (DiskANN index files)
│                   │   ├── faiss-flat
│                   │   │   └── subset_1024_499992.index
│                   │   └── faiss-hnsw
│                   │       └── subset_m16_efc40_1024_499992.index
│                   └── subset_1024_499992.npmmap
└── text
    ├── corpus
    │   └── kilt
    │       ├── bm25_preprocessed
    │       │   └── kilt_w100_title_modified_subset_499992
    │       │       └── preprocessed_corpus.json
    │       ├── doc_id_to_wikipedia_id_mapping.p
    │       ├── kilt_w100_title_modified_subset_499992.tsv
    │       ├── kilt_w100_title.tsv
    │       └── mapping_KILT_title.p
    └── query
        └── kilt
            ├── nq-dev-kilt_100.jsonl
            └── nq-train-kilt_100.jsonl
```

## register index data in config base_settings files

Prepared index data so far can be registered in config files as below. Note that we assume `${data} = ./data` in these example files.

- [retriever](../scripts/configs/base_settings/retrievers.json)
- [corpus](../scripts/configs/base_settings/corpora.json)
- [dataset](../scripts/configs/base_settings/datasets.json)

Congraturations! You are now ready to use RᴀLLᴇ GUI. Please refer to the [instruction](gui_usage.md) or [screencast](https://www.youtube.com/watch?v=ShMNYSBJNnc) for the usage.

