# The Journey <!-- omit in toc -->

This doc serves as a record of the journey when creating this project; including all of our design choices, thought processes, and even our blunders.

## Table of Contents <!-- omit in toc -->
- [Beginning Stages](#beginning-stages)
  - [Tokenization](#tokenization)
- [Back and Fourths on filetype](#back-and-fourths-on-filetype)
  - [The Re-Tokenizing](#the-re-tokenizing)


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