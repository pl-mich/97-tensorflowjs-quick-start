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

### 2.1. Preexisting [Teachable Machine](https://teachablemachine.withgoogle.com/)

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
- Implement a browser-based prediction mechanism and a visualization 
  compoment of it.  

However, [this](https://observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js) 
perhaps can give more details about what is going on under the hood.
