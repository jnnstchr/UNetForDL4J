
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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class SegNetPretrained {
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 1;
    private static int batchSize = 1;

    private static int width = 512;
    private static int height = 512;
    private static int channels = 3;


    @Builder.Default
    private int[] inputShape = new int[]{3, 512, 512};
    @Builder.Default
    private int numClasses = 2;
    @Builder.Default
    private IUpdater updater = new Nesterovs();
    @Builder.Default
    private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public String dataPath;

    public static void main(String[] args) throws IOException {
        SegNetPretrained segNet = new SegNetPretrained();
        segNet.importData();
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
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        System.out.println(pretrainedNet.summary());


//        FCN.setScaler(dataTrainIter, dataTestIter, pretrainedNet);

        ComputationGraph segnetTransfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .setFeatureExtractor("block5_pool")
                .removeVertexAndConnections("predictions")
                .removeVertexAndConnections("fc2")
                .removeVertexAndConnections("fc1")
                .removeVertexAndConnections("flatten")
//                .removeVertexKeepConnections("block5_pool")
                .addLayer("16", new Upsampling2D.Builder(2).build(), "block5_pool")
                .addLayer("17", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(512).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "16")
                .addLayer("18", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(512).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "17")
                .addLayer("19", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(512).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "18")

                .addLayer("20", new Upsampling2D.Builder(2).build(), "19")
                .addLayer("21", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(512).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "20")
                .addLayer("22", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(256).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "21")
                .addLayer("23", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(256).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "22")

                .addLayer("24", new Upsampling2D.Builder(2).build(), "23")
                .addLayer("25", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(256).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "24")
                .addLayer("26", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(128).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "25")

                .addLayer("27", new Upsampling2D.Builder(2).build(), "26")
                .addLayer("28", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(128).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "27")
                .addLayer("29", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                        .padding(1, 1).nIn(64).nOut(1).cudnnAlgoMode(cudnnAlgoMode).build(), "28")
                .addLayer("30", new Upsampling2D.Builder(2).build(), "29")
                .addLayer("31", new ActivationLayer.Builder().activation(Activation.SOFTMAX)
                        .build(), "30")
//                .addLayer("output", new OutputLayer.Builder().nIn(64).nOut(3).build(), "30")
                .addLayer("output", new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID).build(), "31")
                .setOutputs("output")
                .build();


        segnetTransfer.init();
        System.out.println(segnetTransfer.summary());
        segnetTransfer.fit(dataTrainIter, epochs);
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        setScaler(scaler, dataTrainIter, dataTestIter, segnetTransfer);
        segnetTransfer.fit(dataTrainIter, epochs);

        int j = 0;
        while (dataTestIter.hasNext()) {
            DataSet t = dataTestIter.next();
            scaler.revert(t);
            INDArray[] predicted = segnetTransfer.output(t.getFeatures());
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

    static void setScaler(NormalizerMinMaxScaler scaler, DataSetIterator dataTrainIter, DataSetIterator dataTestIter, ComputationGraph pretrainedNet) {
        scaler.fitLabel(true);
        scaler.fit(dataTrainIter);
        dataTrainIter.setPreProcessor(scaler);
        scaler.fit(dataTestIter);
        dataTestIter.setPreProcessor(scaler);
        System.out.println(pretrainedNet.summary());
    }

}
