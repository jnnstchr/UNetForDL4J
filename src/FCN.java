
import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class FCN<numClasses> {
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 1;
    private static int batchSize = 1;

    private static int width = 512;
    private static int height = 512;
    private static int channels = 3;

    public String dataPath;
    public int numClasses = 2;

    public static void main(String[] args) throws IOException {
        FCN fcn = new FCN();
        fcn.importData();
    }
    public void importData() throws IOException {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        dataPath = "/home/jstachera/ekek/Training/deeplearning";
        File trainData = new File(dataPath + "/train/image");
        File testData = new File(dataPath + "/test/image");
        LabelGenerator labelMakerTrain = new LabelGenerator(dataPath + "/train");
        LabelGenerator labelMakerTest = new LabelGenerator(dataPath + "/test");

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);


        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMakerTrain);
        rrTrain.initialize(train, null);

        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMakerTest);
        rrTest.initialize(test, null);

        int labelIndex = 1;

        DataSetIterator dataTrainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, labelIndex, true);
        DataSetIterator dataTestIter = new RecordReaderDataSetIterator(rrTest, 1, labelIndex, labelIndex, true);

        VGG16ImagePreProcessor vgg16ImagePreProcessor = new VGG16ImagePreProcessor();
        dataTrainIter.setPreProcessor(vgg16ImagePreProcessor);
        dataTestIter.setPreProcessor(vgg16ImagePreProcessor);
        ZooModel zooModel = VGG16.builder().build();
        zooModel.setInputShape(new int[][]{{7, 7, 512}});
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        System.out.println(pretrainedNet.summary());
        setScaler(dataTrainIter, dataTestIter, pretrainedNet);
        NormalizerMinMaxScaler scaler;

        ComputationGraph vggTransfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")

                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)
                                .nOut(2)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .setOutputs("predictions")
                .build();


        vggTransfer.init();
        System.out.println(vggTransfer.summary());
        vggTransfer.fit(dataTrainIter, epochs);

        int j = 0;
        while (dataTestIter.hasNext()) {
            DataSet t = dataTestIter.next();
//            scaler.revert(t);
            INDArray[] predicted = vggTransfer.output(t.getFeatures());
            INDArray pred = predicted[0].reshape(new int[]{512, 512});
            Evaluation eval = new Evaluation();

            eval.eval(pred.dup().reshape(512 * 512, 1), t.getLabels().dup().reshape(512 * 512, 1));
            System.out.println(eval.stats());
            DataBuffer dataBuffer = pred.data();
            double[] classificationResult = dataBuffer.asDouble();
            ImageProcessor classifiedSliceProcessor = new FloatProcessor(512, 512, classificationResult);
            //segmented image instance
            ImagePlus classifiedImage = new ImagePlus("pred" + j, classifiedSliceProcessor);
            new File(dataPath + "/predictions/" + j + ".png").mkdirs();
            IJ.save(classifiedImage, dataPath + "/predictions/" + j + ".png");
            j++;
        }
    }

    static void setScaler(DataSetIterator dataTrainIter, DataSetIterator dataTestIter, ComputationGraph pretrainedNet) {
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fitLabel(true);
        scaler.fit(dataTrainIter);
        dataTrainIter.setPreProcessor(scaler);
        scaler.fit(dataTestIter);
        dataTestIter.setPreProcessor(scaler);
        System.out.println(pretrainedNet.summary());
    }
}
