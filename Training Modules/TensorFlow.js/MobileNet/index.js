/*
* Phew! The formatting warnings back in Visual Studio Code was a real headache!
*
* This is already surprisingly similar to the actual Teachable Machine over
* on https://teachablemachine.withgoogle.com/train/image, save for some
*  (major) different in the user interface. It is definitely a major
*  concern, but only of secondary importance.
*
* JUST LOOK AT HOW MUCH COMMENT YOU HAVE BEEN WRITING FOR THIS PIECE OF CRAP!
*/

// Man, you just gotta love those out-of-the-box references to code you have
// no idea how it works...
const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // MEMORIZE THIS getElementById THING!
  // /** Make a prediction through the model on our image. */
  // const imgEl = document.getElementById('img');
  // const result = await net.classify(imgEl);
  // console.log(result);

  /**
   * Create an object from Tensorflow.js data API which could capture image
   * from the web camera as Tensor.
   */
  const webcam = await tf.data.webcam(webcamElement);

  /**
   * Reads an image from the webcam and associates it with a specific
   * class index.
   *
   * Javascript refresher! In C++ this function declaration would more be likely
   * rendered as void knnClassifier::addExample(double classId) const {}
   *
   * Also note that it is an ASYNCHRONOUS function.
   */

  const addExample = async classId => {
    /** Capture an image from the web camera. */
    const img = await webcam.capture();

    /**
     * Run all but the last layer of the model (hence the infer() function call)
     * to get the intermediate activation of MobileNet 'conv_preds' and
     * pass that to the KNN classifier.
     */
    const activation = net.infer(img, true);

    /**
     * Pass the intermediate activation to the classifier.
     *
     * That's right! This is it! No rebuilding or retraining models,
     * no callbacks and all those bullshit
     */
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  /*
  * When clicking a button, add an example for that class.
  *
  * This is how addEventListener() works:
  * When the event defined by the first parameter occurs,
  * Execute the function defined in the latter parameter
  */
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  // Good ol' infinite loop
  while (true) {

    // If the classifier is indeed trained...
    if (classifier.getNumClasses() > 0) {

      const img = await webcam.capture();

      /** Get the activation from mobilenet from the webcam. */
      const activation = net.infer(img, 'conv_preds');

      /**
       * Get the most likely class and confidence from the classifier
       * module.
       *
       * See how the predictClass method takes in the ACTIVATION as its
       * parameter?
       */
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C', 'noAction'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();

    }

    // Code for making predictions on the webcam input image with the
    // MobileNet classifier straight out-of-the-box
    // const img = await webcam.capture(); // Capture image from the webcam
    // const result = await net.classify(img);
    //
    // /*
    // * This is NOT the real "console" in the F12 window, only a <div>
    // * container.
    // * See how the innerText attribute is called to modify contents in a
    // * <div> container.
    // *
    // * There's no way I'm supposed to do this in C++ or Java.
    // */
    // document.getElementById('console').innerText = `
    //   prediction: ${result[0].className}\n
    //   probability: ${result[0].probability}
    // `;
    //
    // // Dispose the tensor to release the memory.
    // img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire. Note that the "next animation frame" does NOT mean the next
    // frame in the webcam feed. Rather, this function waits for the BROWSER
    // to update the window animation.
    await tf.nextFrame();
  }

}

app();
