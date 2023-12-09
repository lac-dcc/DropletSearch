# Droplet Search

Droplet Search is a technique to optimize machine learning kernels, based on the coordinate descent algorithm. This algorithm is currently part of Apache TVM (For more details, see this <a href="https://github.com/apache/tvm/pull/14683">PR</a>). To know more about it, you can take a look into [this paper](https://homepages.dcc.ufmg.br/~michaelcanesche/paper/droplet_paper.pdf).

<p align="center">
  <img alt="logo" src="./docs/DropletSearch.png" width="30%" height="auto"/><br>
  <a href="https://github.com/lac-dcc/garkain/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPL%203.0%20only-green?style=for-the-badge" alt="License: GPL v3"></a>
  <a href="https://github.com/PyCQA/pylint"><img src="https://img.shields.io/badge/linting-pylint-yellowgreen?style=for-the-badge" alt="Linting: Pylint"></a>
  <a href="https://github.com/lac-dcc/garkain/commits/main">
    <img src="https://img.shields.io/github/last-commit/lac-dcc/garkain/main?style=for-the-badge"
         alt="Last update">
  </a>
</p>

## **Table of Contents**

* [Introduction](#introduction)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Setup](#setup)
    * [Running](#running)
* [Structure](#structure)

---
<a id="introduction"></a>

## **Introduction**

Droplet is [merged](https://github.com/apache/tvm/pull/14683) on Apache TVM since version 0.13.0. This repository is used as an artifact for the paper.

---
<a id="getting-started"></a>

## **Getting Started**

In this section are the steps to reproduce our experiments.

### **Prerequisites**

You need to install the following packages to run this project:

* [Docker](https://www.docker.com/get-started/) and [Docker Compose](https://docs.docker.com/compose/install/) to run our experiments
* [Python-3](https://www.python.org/downloads/) to plot the results in the project's Jupyter Notebook

For ***nvidia docker***, please follow these instructions: [Nvidia Container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

<a id="setup"></a>

###  **Setup**

We developed a dockerfile with the experiments and all requirements installed. We recommend using this solution if you want to compare it with our solution. Below, for each architecture supported, is presented how to build the docker.

```
bash scripts/build_docker.sh <ARCH>
```

Where ```<ARCH>``` can be x86, arm, or cuda.

<a id="running"></a>

### **Running**
You can run the docker following command line:

```bash
bash scripts/run_docker.sh <ARCH>
```

Where ```<ARCH>``` can be x86, arm, or cuda.

To execute the neural networks models (Figure 11):

```bash
bash scripts/cnn_models.sh <ARCH>
```

To measure the impact of the p-value in the droplet (Figure 12):
```bash
bash scripts/droplet_pvalue.sh <ARCH>
```

To execute microkernels (Appendix), you must use the following script:

```bash
bash scripts/microkernels.sh
```


---
<a id="structure"></a>

## Structure
The repository has the following organization:

```bash
|-- results: "Place which your data will be saved for the default"
|-- docker: "Scripts for building the docker"
|-- docs: "Repository documentation"
|-- scripts: "Scripts for building the docker and generating some images"
|-- src: "Source code"
    |-- handmade: "Extra experiments using the droplet to verify how the space search works"
    |-- microkernels: "Python scripts to run microkernel presents in the paper"
    |-- tvm: "Python scripts to run NN models presented in the paper"
|-- thirdparty: "Third-party code for comparison with our experiments."
```
