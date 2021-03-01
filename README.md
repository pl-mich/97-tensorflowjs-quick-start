> Forked from Angular Firebase for TFJS starter app (BlueML)

# "TRUST THE PLAN"

> Do I sound like I believe in QAnon bullshit? Hello, FBI. I never did.

## Short-term Goals

- [ ] Go through the TensorFlow.JS training modules
- [ ] Go through the TensorBoard training modules
- [ ] Build a Fashion MNIST image classifying module OR grab an existing 
  image classification module from the repository 
  [tensorflow/tfjsmodels](https://github.com/tensorflow/tfjs-models)
- [ ] Construct a TensorBoard logging applet out of it
- [ ] Publish the applet through [TensorBoard.dev]

## For the Next Few Months...

- [ ] Go through source code of [tensorflow/tfjsmodels](https://github.com/tensorflow/tfjs-models)
  and Google Teachable Machine
- [ ] Find out ways to embed TensorBoard.dev visualizations into webpages
- [ ] Any chance of manually writing the TensorFlow log files from 
  preexisting data? 
- [ ] Any chance of presenting the graph only and NOT the potentially 
  sensitive data?
- [ ] Build a neat little in-house Python package for generating a 
  visualization module from an existing model and a JavaScript applet to 
  produce a webpage.  
- [ ] Try out more complex and realistic data generated from other SOCR projects

# [Learning Materials](https://docs.google.com/document/d/1T3_WfTBotKqgHlf5A3j70_5WLIWBtmWB-Sk0IhAYULk/edit)

## Useful Links

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
- [TensorBoard Training Modules](https://www.tensorflow.org/tensorboard/get_started)
- [tensorflow/tensorboard repository](https://github.com/tensorflow/tensorboard) 

### Tensorflow.js

- [https://js.tensorflow.org/](https://js.tensorflow.org/)
- [Fireship Tutorial](https://fireship.io/lessons/tensorflow-js-quick-start/)  
- [freeCodeCamp.org video tutorial](https://www.youtube.com/watch?v=EoYfa6mYOG4)
- [Converting an existing Keras model into a TensorFlow.js model](https://www.tensorflow.org/js/tutorials/conversion/import_keras)

### Convolutional Neural Networks

- [Stanford CS 231N](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [3Blue1Brown video](https://www.youtube.com/watch?v=aircAruvnKk)

## Vegi's Proposed Approach

1. I believe Dimensionality reduction and Tensorboard 
   might not be your first priority. 
   Prof. Dinov's students have done a fair bit of work on Dimensionality 
   Reduction and you have my initial TensorBoard work on SOCR.
   
2. The primary goal for your team in my opinion should be TensorFlow.js,
   i.e., you should leverage TensorFlow.js framework to do
   data processing, training, and prediction right in the 
   browser without any standalone server.
   
3. This is how I would lay out the work.
   - *Level 1*: Learn Image Classification Algorithms. 
     Implement a web app where a user uploads an image on your web app and 
     your model predicts. For this, you can start your work on 2D ABIDE images.
   - *Level 2*: Now train the model on 3D images. Enable your web app to 
     facilitate uploading and data processing of 3D images.
   - *Level 3*: Enrich each image with complimentary features such as 
     Age, sex, demographics etc. 
     Now the user can choose to get the predicts solely based on images or 
     can get more accurate predictions with complimentary features 
     he inputs on the web app.


# Tensorflow.js MNIST Angular Demo

This demo imports an MNIST ConvNet trained in Keras Python, then makes predictions with TensorFlow.js

- clone it, cd into it, `npm install && ng serve`

## Use a Different Keras Model

```bash
tensorflowjs_converter --input_format keras keras/yourWeights.h5 src/assets
```
