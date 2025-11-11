# The Journey <!-- omit in toc -->

This doc serves as a record of the journey when creating this project; including all of our design choices, thought processes, and even our blunders.

## Table of Contents <!-- omit in toc -->
- [Beginning Stages](#beginning-stages)
  - [Tokenization](#tokenization)
- [Back and Fourths on filetype](#back-and-fourths-on-filetype)
  - [The Re-Tokenizing](#the-re-tokenizing)
  - [Model Collapse](#model-collapse)
  - [`Dataset` \& `Dataloader` Refreshers](#dataset--dataloader-refreshers)


## Beginning Stages

- garner massive password dataset
- hash dataset using md5
- massive speedbump on comprehending tokenization and how it would relate/make the most sence with our unique case

### Tokenization
- currently two different vocabs (10/13)
  - one for hashes and another for characters
  - thought process:
    - the hex representation of a number doesn't have the same meaning as a traditional character
    - didn't want to present them to the model as having similar meaning
  

## Back and Fourths on filetype

- previously burned with massive overhead and memory usage in past LLM project that frequently crashed due to OOM errors
- moved filetype to YAML for early preprocessing for easier readibility
- landed on JSON filetype for training
  - sharded 1MIL hash -> pw pairs into 10 JSON shards
  - later realized JSON was overkill/unnecessary
  - refactored all shards to TSV


### The Re-Tokenizing

- once I leared of pyTorch's Dataset class, I realized that all of my tokenization research and code wasn't necessary
  - with this class, the tokenization happens when you load the trainig session
  
>NOTE: just learned about `.npz` files.  
> - might, once more, change the file type of the shards to `.npz` so all of the data is prepocessed as numpy arrays so training is even faster.
> - will probably wait to see how fast/slow a training session is now w/ the TSV's before spending the time on it 

### Model Collapse

In the models [prediction file](../model/predictions/), I soon realized every prediction started with a `1`.

After investigation, `1` was the most common first character at 8%, but that still shouldn't cause complete model collapse.

I was using __teacher forcing__ in [`model/trainer.py`](../model/trainer.py)--`line 523: logits = self.model(hashes, pw)`, which is acceptable practice for computing loss, but for predictions/evaluation, I should use autogregressive generation where it generates one token at a time using its own previous predictions.

>The problem is that greedy decoding with teacher forcing doesn't represent real-world inference. The model never learns to correct itself because it's always fed the correct sequence during both training AND evaluation!
>
>root cause: The model hasn't learned to generate coherently because it's always given the _correct_ context. When you look at predictions via `argmax`, you're seeing what it predicts at each position GIVEN the correct
>prefix—but the model is biased toward predicting '1' when it doesn't have good signal from the hash.


### `Dataset` & `Dataloader` Refreshers

[Building a Neural Network with PyTorch in 15 Minutes | Coding Challenge](https://www.youtube.com/watch?v=mozBidd58VQ)  
[PyTorch DataLoader Explained: How to make Basic and Custom Datasets](https://www.youtube.com/watch?v=7tfGVOulJRM)   
[PyTorch Tutorial 09 - Dataset and DataLoader - Batch Training](https://www.youtube.com/watch?v=PXOzkkB5eH0&t=491s)  
[Build Your First Pytorch Model In Minutes! [Tutorial + Code]](https://www.youtube.com/watch?v=tHL5STNJKag&t=411s)