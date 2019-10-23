import { MnistData } from "./data.js";
console.log("teste");
console.log(tfvis);
console.log(tf);

async function run() {
  console.log(tf);
  console.log("chegou");
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  const model = await getModel();
  
  tfvis.show.modelSummary({ name: "Model Architecture" }, model);
  await showAccuracy(model, data);
  await showConfusion(model, data);

  
}

async function showExamples(data) {
  console.log("showexemple");
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}
//Transferencia de aprendizado
async function getModel() {
  const model = tf.sequential(); //cria uma interface entre camadas , pilar da tranferencia de aprendizado

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // Na primeira camada de nossa rede neural convolucional, temos
  // para especificar a forma da entrada. Em seguida, especificamos alguns parâmetros para
  // a operação de convolução que ocorre nesta camada.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling"
    })
  );

  // A camada MaxPooling atua como uma espécie de amostragem descendente usando valores máximos
  // em uma região em vez de calcular a média.
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Repita outra pilha conv2d + maxPooling.
  // Note que temos mais filtros na convolução.
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling"
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Agora, nivelamos a saída dos filtros 2D em um vetor 1D para preparar
  // para entrada em nossa última camada. Essa é uma prática comum ao alimentar
  // dados dimensionais mais altos para uma camada de saída de classificação final.
  model.add(tf.layers.flatten());

  // Nossa última camada é uma camada densa que possui 10 unidades de saída, uma para cada
  // classe de saída (ou seja, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax"
    })
  );

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

async function train(model, data) {
  //acompanahamento de métricas
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Model Training",
    styles: { height: "1000px" }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });

}

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine"
];

async function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1
  ]);
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = await doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = await doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(
    container,
    { values: confusionMatrix },
    classNames
  );

  labels.dispose();
}

run();
