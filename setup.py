# Copyright (c) Kioxia corporation and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ralle",
    version="0.1.0",
    description="Retrieval-Augmented Large Language model Evaluation tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "gradio",
        "mlflow"
    ],
)