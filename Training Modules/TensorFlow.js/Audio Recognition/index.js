// Global variables and constants definition
let recognizer;

/** Storage for all training data */
let examples = [];

/**
 * We are using samples that are 3 frames long (~70ms samples, one frame is
 * ~23ms of audio) since we are making sounds instead of speaking whole
 * words to control the slider.
 */
const NUM_FRAMES = 3;

/**
 * The model has 4 layers:
 * a convolutional layer that processes the audio data
 * (represented as a spectrogram),
 * a max pool layer, a flatten layer, and
 * a dense layer that maps to the 3 actions.
 */
let model;

/**
 * Each frame is 23ms of audio containing 232 numbers that correspond to
 * different frequencies
 * (232 was chosen because it is the amount of frequency buckets needed to
 * capture the human voice)
 */
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];


// Helper functions
/**
 * And to avoid numerical issues, we normalize the data
 * to have an average of 0 and a standard deviation of 1.
 * In this case, the spectrogram values are usually
 * large negative numbers around -100 and deviation of 10.
 */
function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

async function moveSlider(labelTensor) {
  const label = (await labelTensor.data())[0];
  document.getElementById('console').textContent = label;
  if (label == 2) {
    return;
  }
  let delta = 0.1;
  const prevValue = +document.getElementById('output').value;
  document.getElementById('output').value =
    prevValue + (label === 0 ? -delta : delta);
}


// Primary components of the runner code
/** Associates a label with the output of recognizer.listen() */
function collect(label) {
  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }

  /*
  * Since includeSpectrogram is true,
  * recognizer.listen() gives the raw spectrogram (frequency data)
  * for 1 sec of audio, divided into 43 frames,
  * so each frame is ~23ms of audio.
  */
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {

    /**
     * Since we want to use short sounds instead of words to control the
     * slider, we are taking into consideration only the last 3 frames
     * (~70ms)
     */
    let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));

    /*
    * Each training example will have 2 fields:
    * label: 0, 1, and 2 for "Left", "Right" and "Noise" respectively
    * vals: 696 numbers holding the frequency information (spectrogram)
    *
    * I'd assume this works similarly to the push_back() function in C++?
    */
    examples.push({vals, label});
    document.querySelector('#console').textContent =
      `${examples.length} examples collected`;

  }, {
    overlapFactor: 0.999, // What is this?
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true // What's the callback?
  });
}

/** Let the model predict what word is being spoken in real time */
function predictWord() {

  /** Array of words that the recognizer is trained to recognize. */
  const words = recognizer.wordLabels();

  // That's some pretty crappy formatting
  recognizer.listen(({scores}) => {
      // Turn scores into a list of (score,word) pairs.
      scores = Array.from(scores).map((s, i) =>
        ({score: s, word: words[i]}));
      // Find the most probable word.
      scores.sort((s1,
                   s2) => s2.score - s1.score);
      document.querySelector('#console').textContent =
        scores[0].word;
    },
    {probabilityThreshold: 0.75});
}

/** Defines the model architecture */
function buildModel() {
  model = tf.sequential();
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 8, // What is this?
    kernelSize: [NUM_FRAMES, 3], // What about the three here?
    activation: 'relu',
    inputShape: INPUT_SHAPE
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

  // We compile our model to get it ready for training
  /**
   * Refer to https://arxiv.org/abs/1412.6980 for original paper on the adam
   * optimizer.
   *
   * (Can't believe this crap was invented when I was still living in Ithaca!)
   */
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
}

/** Trains the model using the collected data */
async function train() {
  toggleButtons(false);
  const ys = tf.oneHot(examples.map(e => e.label), 3);
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      // Hey! There are logs being outputted! Try doing something with
      // TensorBoard
      onEpochEnd: (epoch, logs) => {
        document.querySelector('#console').textContent =
          `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}

/**
 * Listens to the microphone and makes real time predictions.
 *
 * The code is very similar to the collect() method, which normalizes the raw
 * spectrogram and drops all but the last NUM_FRAMES frames.
 * The only difference is that we also call the trained model
 * to get a prediction.
 */
function listen() {
  // If the recognizer has been listening, stop listening and
  // enable toggling the buttons for the next iteration.
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listen').textContent = 'Listen';
    return;
  }

  toggleButtons(false);
  document.getElementById('listen').textContent = 'Stop';
  document.getElementById('listen').disabled = false;

  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    // Normalize input values
    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));

    /**
     * A Tensor of shape [1, numClasses] representing a probability
     * distribution over the number of classes.
     *
     * More simply, this is just a set of confidences for each of the
     * possible output classes which sum to 1.
     * The Tensor has an outer dimension of 1 because
     * that is the size of the batch (a single example).
     */
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);

    // Make a predictions
    const probs = model.predict(input);

    /**
     * To convert the probability distribution to a single integer
     * representing the most likely class, we call probs.argMax(1), which
     * returns the class index with the highest probability.
     * We pass a "1" as the axis parameter because
     * we want to compute the argMax over the last dimension, numClasses.
     */
    const predLabel = probs.argMax(1);

    /*
    * moveSlider() decreases the value of the slider if the label is 0 ("Left"),
    * increases it if the label is 1 ("Right")
    * and ignores if the label is 2 ("Noise").
    */
    await moveSlider(predLabel);

    /*
    * To clean up GPU memory it's important for us to manually call
    * tf.dispose() on output Tensors.
    * The alternative to manual tf.dispose() is wrapping function calls in a
    * tf.tidy(), but this cannot be used with async functions.
    */
    tf.dispose([input, probs, predLabel]);
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}


async function app() {
  console.log('Loading Speech Recognition model...');
  // How is this crap supposed to work, anyways?
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  console.log('Successfully loaded model');
  buildModel();
  // predictWord();
}

app();
