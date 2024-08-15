---
title: Mnist4All
type: series
---

Mnist4All is a series where I implement an [MNIST](https://en.wikipedia.org/wiki/MNIST_database) classifier neural net in various languages, 
libraries, and with different levels of abstraction. 

1. Python - built on `numpy` - closely follows my mental model for deep learning -
 outlines the base structure which other implementations will reproduce.

2. C - only using standard library - implements tensor from scratch - zero-copy reshape - it's *a lot* slower than `numpy`
