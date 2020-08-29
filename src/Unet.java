
import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import lombok.Builder;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;


public class Unet {
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 3;
    private static int batchSize = 1;

    private static int width = 512;
    private static int height = 512;
    private static int channels = 3;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public String dataPath;

    public static void main(String[] args) throws IOException {
        Unet unet = new Unet();
        unet.importData();
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
        ZooModel zooModel = UNet.builder().build();

        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        System.out.println(pretrainedNet.summary());
        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .setFeatureExtractor("conv2d_23")
//                .removeVertexKeepConnections("activation_23")
//                .removeVertexKeepConnections("conv2d_23")
//                .removeVertexKeepConnections("dropout_1")
//                .removeVertexAndConnections("activation_22")
//                .removeVertexKeepConnections("batch_normalization_22")
//                .addLayer("conv9-2", new ConvolutionLayer.Builder(3,3).stride(1,1).nIn(32).nOut(32)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv2d_22")
//                .addLayer("conv9-3", new ConvolutionLayer.Builder(3,3).stride(1,1).nIn(32).nOut(2)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv9-2")
//                .addLayer("conv10", new ConvolutionLayer.Builder(3,3).stride(1,1).nIn(2).nOut(1)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.IDENTITY).build(), "conv9-3")
                .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "activation_23")
                .setOutputs("output")
                .build();


        unetTransfer.init();
        System.out.println(unetTransfer.summary());
        unetTransfer.fit(dataTrainIter, epochs);

        int j = 0;
        while (dataTestIter.hasNext()) {
            DataSet t = dataTestIter.next();
//            scaler.revert(t);
            INDArray[] predicted = unetTransfer.output(t.getFeatures());
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
    public String toString(ComputationGraph uNet){
        return uNet.summary();
    }
}