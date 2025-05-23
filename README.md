# Compare Schedule-Free Training with Learning Rate Schedulers

This repository is dedicated to an experimental project that compares commonly used learning rate schedulers with a schedule-free method where the learning rate remains fixed during training. The goal is to investigate the practical implications and performance differences between these two approaches in deep learning training.

## Overview

In modern deep learning training, both learning rate schedulers and schedule-free methods typically require a warmup phase to stabilize the optimization process. This project explores how these methods perform under similar training conditions and provides a flexible framework for custom training setups.

## Background

Prior research, particularly the study titled *"Increasing Both Batch Size and Learning Rate Accelerates Stochastic Gradient Descent"*, has shown that combining increased batch sizes with appropriate learning rate strategies can significantly enhance training efficiency. Inspired by these findings, this repository enables custom configurations that allow researchers and practitioners to experiment with varying batch sizes, learning rate schedules, and warmup techniques.

## Features

* Comparative framework for scheduled vs. schedule-free learning rate strategies
* Support for warmup phases in both approaches
* Integration of scalable batch size training as per recent research
* Modular design for easy customization and extension

