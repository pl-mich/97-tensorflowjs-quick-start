# A Few Notes on TensorFlow.js

## 1. [ml5.js](https://learn.ml5js.org/#/)

Would this be a shortcut to all the bullcrap with JavaScript or TensorFlow.js?

I wonder how well it integrates with TensorBoard...

## 2. Transfer Learning 

Sophisticated deep learning models have millions of parameters (weights) 
and training them from scratch often requires large amounts of data of 
computing resources. 
Transfer learning is a technique that shortcuts much of this by taking a 
piece of a model that has already been trained on a related task and 
reusing it in a new model.

Most often when doing transfer learning, we don't adjust the weights of the 
original model. Instead, we remove the final layer and train a new (often 
fairly shallow) model on top of the output of the truncated model.

### 2.1. More Conceptual Details

The activations of the pretrained MobileNet model informally represent 
high-level semantic features of the image that the model has learned.
What we do is feed an image through MobileNet and find other examples 
in the dataset that have similar activations to this image.

In this MobileNet model, the last layer is a softmax normalization function.
Intuitively, this function "squashes" a vector of unnormalized predictions, 
generating a probability for each of the 1000 classes (normalized predictions).

### 2.2 Organizational Hierachies

[MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) 
is trained on [ImageNet](http://www.image-net.org/about-overview)

ImageNet is an image dataset organized according to the WordNet hierarchy. 
Each meaningful concept in WordNet, possibly described by multiple words or 
word phrases, is called a "synonym set" or "synset". 
There are more than 100,000 synsets in WordNet, majority of them are nouns 
(80,000+). In ImageNet, we aim to provide on average 1000 images to 
illustrate each synset. 
Images of each concept are quality-controlled and human-annotated.

[WordNet®](http://wordnet.princeton.edu) is a large lexical database of 
English. Nouns, verbs, adjectives and adverbs are grouped into sets of 
cognitive synonyms (synsets), each expressing a distinct concept. 
Synsets are interlinked by means of conceptual-semantic and 
lexical relations.

### 2.3. Preexisting [Teachable Machine](https://teachablemachine.withgoogle.com/)

~~This is probably not the best option for a new proprietary interface, 
but is definitely worth the effort taking a look into.~~

Still, that's some quite impressive visualization implementation! I wonder 
whether that was plain old TensorBoard or some other crazy JavaScript crap 
is going on...

HOLY SCHITT! That is really some cool stuffs going on!

Now that is your objective, your goal of sorts...

- A proprietary interface that can be easily accessed by other SOCR projects
- Keeps users' data in their local machines, use their machines to train a 
  model, while keeping a copy of that model in the cloud.
- Implement a browser-based prediction mechanism and a 
  visualization compoment of it.  


### Under the hood

[This](https://observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js) 
perhaps can give more details about what is going on under the hood.

Essentially, it is just a bunch of vector peoducts and matrix multiplications.
