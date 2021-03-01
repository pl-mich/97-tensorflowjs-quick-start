console.log('Hello TensorFlow')

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  // Fetch data from a remote source.
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')  
  // Convert the data into a JSON format
  const carsData = await carsDataResponse.json()
  // const dataString = JSON.stringify(carsData)
  // console.log(dataString)

  /*
   * Map the values from car to new attributes with different names and
   * Drop contents from car that do not satisfy the condition that neither mpg
   * nor horsepower is null
   *
   * Map takes in a function as a parameter
   * Filter takes in a function as a parameter
   *
   * => passes car into a function that returns the variables in the brackets 
   */
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower
  })).filter(car => (car.mpg != null && car.horsepower != null));
  return cleaned;
}

function createModel () {
  // Create a sequential model
  const model = tf.sequential(); 

  // Add a single input layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling 
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    // If you want to return a f**kton of data, use a JS object!
    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  })
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    // sgd is more intuitive
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  const batchSize = 32;
  const epochs = 50;

  // THIS IS ASYNCHRONOUS!
  return await model.fit(inputs, labels, {
    batchSize, // don't define as kwargs!
    epochs,
    shuffle: true,
    // Visualize training performance with loss and mse
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  })
}

// THIS IS A SYNCHRONOUS FUNCTION!
function testModel (model, inputData, normalizationData) {
  // normalizationData generated from convertToTensor
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    // linspace returns a tensor, not a list
    // ... quite unlike in MATLAB
    const xs = tf.linspace(0, 1, 100)
    // Note that the tensor needs to have a similar shape
    // ([num_examples, num_features_per_example]) as when we did training.
    const preds = model.predict(xs.reshape([100, 1]))

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

    /*
     * Un-normalize the data
     * .dataSync() is a method we can use to get a typedarray of the
     * values stored in a tensor. This allows us to process those values in
     * regular JavaScript. This is a synchronous version of the .data() method
     * which is generally preferred.
     */
    return [unNormXs.dataSync(), unNormPreds.dataSync()]
  })

  // Format the original and the predicted data into objects with
  // the same properties to be plotted in one graph
  // Note how "x" and "y" correspond exactly to the scatterplot axes
  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }))

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted']
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}

async function run () {
  // Load and plot the original input data that we are going to train on.
  // Wait for the previous function to execute
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }))
  // let csvContents = "data:text/csv;charset=utf-8," 
  //   + values.map(d => d.join(",")).join("\n");
  // var encodedUri = encodeURI(csvContent);
  // window.open(encodedUri);
  
  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )

  // More code will be added below
  // Create the model
  const model = createModel()
  tfvis.show.modelSummary({ name: 'Model Summary' }, model)
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data) // This is an object
  const { inputs, labels } = tensorData // Read values of properties from the object

  // Train the model
  await trainModel(model, inputs, labels) // ASYNCHRONOUS
  console.log('Done Training')

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData)
}

// Used to run the entire thing?
document.addEventListener('DOMContentLoaded', run)
