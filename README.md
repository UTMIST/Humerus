# humerus-bot

`Brian's Contribution`

## Introduction

The humerus bot is a NLP project designed to *win* Cards Against Humanity.


This repository contains code for the [humerus bot](http://humerusdecks.com/#new) as well as artefacts/code for prior unsuccessful approaches.



## Background Information

Cards Against Humanity is a popular humor-based party game where one attempts to select the funniest response (white card) to a prompt (black card).


### Client

An existing cards against humanity implementation, [massive decks](https://github.com/Lattyware/massivedecks) was forked and modified for game data collection and to integrate with our model backend.
This is package and deployed with docker on aws.

We used GPT-2 to generate some cards for use [link to cards, notebook] but they ultimately did not make enough sense for a reasonable game experience.


@Matthew things to add?


### Server/Model

#### Causual Language Modelling Approach

Assuming that large causual language models e.g. GPT-2 can reasonably model text semantics and meanings (at least superficially), one may use them to determine a "funnyness score"


@ Alston, can you extend on this?

 To mention:
1. how it works
2. mention GLTR [source below] because we are basically implementing that
3. proposed weighted log sum for funnyness score
4. why it didn't work: resource constraints

#### Simple NN
With resource constraints in mind we then implemented a more lightweight model. 

[link to notebooks]
1. Possible plays (white card + black card combination )are embedded with BERT and then PCA'd. 69 dimensions turned out to be a good number.
2. As pre-generating all embeddings for all plays would take up too much memory, it is assumed that the PCA matrix will not vary significantly for other plays and so is saved for on-the-fly embedding in deployment.
2. A simple two-layer dense predictive neural net is trained on collected game data. 
3. Networks is deployed via a simple flask API [link]


// @ Jack: Talk a bit about PCA, put some figures in?


### Challenges 
- model design choices: considering deployment, cost, runtime, etc
- Architecture (front/backend)
- What makes something funny? 


## What's next 
- benchmark model


## References

https://cpury.github.io/ai-against-humanity/

GLTR: http://gltr.io/
GLTR Paper: https://arxiv.org/abs/1906.04043


### Team Members

```json5
{
        'Alston Lo' : "https://github.com/alstonlo",
        'Brian Chen' : "https://github.com/ihasdapie",
        'Matthew Ao' : "https://github.com/aoruize",
        'Jack Cai' : "https://github.com/caixunshiren",
        'Zoha Rehan' : "https://github.com/zoharehan"
}

```




































