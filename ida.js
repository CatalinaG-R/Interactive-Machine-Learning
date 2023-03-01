import '@marcellejs/core/dist/marcelle.css';
import {
  dashboard,
  imageUpload,
  confidencePlot,
  select,
  text,
  imageDisplay,
  dataStore,
  datasetBrowser,
  button,
  dataset,
  textInput,
  mobileNet,
  tfjsModel,
  slider
} from '@marcellejs/core';
import { gradcam } from './components';
import imagenet_labs from './imagenet_class_index.json';
import model from '../public/mobilenet_v2/model.json';
import { preprocessImage } from './preprocess_image';
import { umap } from './components/umap';



const featureExtractor = mobileNet();

const label = textInput();
label.title = 'Enter your genre';
const labelButton = button('add your own genre');
labelButton.title ='Add the genre';
const capture = button('Hold to record instances');
capture.title = 'Capture instances t o the training set';


const store = dataStore('localStorage');
const trainingSet = dataset('training-set-umap', store);
const trainingSetBrowser = datasetBrowser(trainingSet);


// -----------------------------------------------------------
// UMAP
// -----------------------------------------------------------

const trainingSetUMap = umap(trainingSet);

const updateUMap = button('Update Visualization');
updateUMap.$click.subscribe(() => {
  trainingSetUMap.render();
});

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const labels = Object.values(imagenet_labs).map((x) => x[1]);
const input = imageUpload({ width: 224, height: 224 });

const classifier = tfjsModel({
  inputType: 'generic',
  taskType: 'classification',
});
classifier.loadFromUrl(model);  // THIS IS THE MODEL !!! USE OURS 

classifier.labels = labels;

// -----------------------------------------------------------
// SINGLE IMAGE PREDICTION
// -----------------------------------------------------------

const gc = gradcam();

classifier.$training.subscribe(({ status }) => {
  if (status === 'loaded') {
    gc.setModel(classifier.model);
    gc.selectLayer();
  }
});

const topK = 10;
const $inputStream = input.$images
  .map(preprocessImage({ preset: 'keras:mobilenet_v2' }))
  .awaitPromises();
const $predictions = $inputStream
  .map(async (img) => classifier.predict(img))
  .awaitPromises()
  .map(({ label, confidences }) => {
    const conf = Object.entries(confidences)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);
    return {
      label: labels[parseInt(label, 10)],
      confidences: conf.reduce((x, y) => ({ ...x, [labels[parseInt(y[0])]]: y[1] }), {}),
    };
  });

const selectClass = select(labels);

$predictions.subscribe(({ label, confidences }) => {
  selectClass.slider.$options.set(Object.keys(confidences));
  selectClass.slider.$value.set(label);
});

const plotResults = confidencePlot($predictions);

const gcDisplay = [
  imageDisplay(input.$images),
  imageDisplay(
    $predictions
      .merge(selectClass.$value)
      .sample($inputStream)
      .map((img) => gc.explain(img, labels.indexOf(selectClass.$value.get())))
      .awaitPromises(),
  ),
];

input.$images
  .filter(() => capture.$pressed.get())
  .map(async (x) => ({
    x: await featureExtractor.process(x),
    y: label.$value.get(),
    thumbnail: input.$thumbnails.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSet.create);

const dash = dashboard({
  title: 'Book Genre',
  backgroundColor: 'black',
  author: 'Marcelle Pirates Crew'
});

const slid = slider({
  values: [5],
  min: 0,
  max: 5,
  step: 1,
});


const buttonx = text(
  `<button style="background-color: black; color: #FFFFFF; font-size: 24px; padding: 16px 130px; border: none; border-radius: 8px;" onclick="window.location.href='http://localhost:5173/#page'">Start</button>`
);
buttonx.title='Upload your book';

const t = text(
  `<img src="https://i.postimg.cc/c4zS92YM/Screenshot-2023-02-23-at-1-40-45-PM.png width="150"/>`,
);
t.title = 'Find out what genre your book is';

const t1 = text (
  `<style> .genres-container {
    display: flex;
    flex-direction: column;
    justify-content: left;
    align-items: left;
    margin-bottom: 20px;
  }
  
  .genre-button {
    background-color: black;
    color: white;
    font-size: 16px;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    margin-bottom: 10px;
    cursor: pointer;
  }
  
  .genre-button:hover {
    background-color:  white;
  }</style><div class="genres-container">
  <button class="genre-button" data-genre="Fiction">Action</button>
  <button class="genre-button" data-genre="Action">Comedy</button>
  <button class="genre-button" data-genre="Drama">Drama</button>
  <button class="genre-button" data-genre="Comedy">Sci-Fi</button>
  <button class="genre-button" data-genre="Non-fiction">Action</button>
  <button class="genre-button" data-genre="Mystery">Comedy</button>
  <button class="genre-button" data-genre="Horror">Drama</button>
  <button class="genre-button" data-genre="Romance">Sci-Fi</button>
  <button class="genre-button" data-genre="Fantasy">Drama</button>
  <button class="genre-button" data-genre="Science fiction">Sci-Fi</button>
</div>`,
)
t1.title = 'Different genres';

const listOfImages = []; // Array with 5 images, need to add the images from the dataset
let genreImage = imageDisplay();
slid.$values.subscribe((x) => {
  genreImage = imageDisplay(listOfImages[x])
});
genreImage.title = 'Books with the same genre';

slid.title = 'Books with the same genre';
dash.page('Main').sidebar(buttonx,t1).use(t);
dash.page('Page').sidebar(input, classifier,slid,genreImage,).use( plotResults, selectClass, gcDisplay);
dash.settings.use(trainingSet);

dash.show();
