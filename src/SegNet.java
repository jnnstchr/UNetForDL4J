
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
import org.deeplearning4j.nn.weights.WeightInit;
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

public class SegNet {
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
        SegNet segNet = new SegNet();
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
        NormalizerMinMaxScaler scaler;

        FCNnoTransfer fcNnoTransfer = new FCNnoTransfer();
        ComputationGraph cp = fcNnoTransfer.init();

        setScaler(dataTrainIter, dataTestIter, cp);
        cp.fit(dataTrainIter, epochs);

        int j = 0;
        while (dataTestIter.hasNext()) {
            DataSet t = dataTestIter.next();
//            scaler.revert(t);
            INDArray[] predicted = cp.output(t.getFeatures());
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

    public ComputationGraphConfiguration computationGraph() {
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .activation(Activation.RELU)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .graphBuilder()
                        .addInputs("in")
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build(), "in")
                        .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "0")
                        .layer(2, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "1")
                        // block 2
                        .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "2")
                        .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "3")
                        .layer(5, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "4")
                        // block 3
                        .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "5")
                        .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "6")
                        .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "7")
                        .layer(9, new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "8")
                        // block 4
                        .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "9")
                        .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "10")
                        .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "11")
                        //block 5
                        .layer(13, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "12")
                        .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "13")
                        .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "14")
                        //upsample
                        .layer(16, new Upsampling2D.Builder(2).build(), "15")
                        .layer(17, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                            .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "16")
                        .layer(18, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "17")
                        .layer(19, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "18")

                        .layer(20, new Upsampling2D.Builder(2).build(), "19")
                        .layer(21, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "20")
                        .layer(22, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "21")
                        .layer(23, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "22")

                        .layer(24, new Upsampling2D.Builder(2).build(), "23")
                        .layer(25, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "24")
                        .layer(26, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "25")

                        .layer(27, new Upsampling2D.Builder(2).build(), "26")
                        .layer(28, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "27")
                        .layer(29, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "28")
                        .layer("30", new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX).build(), "29")
                        .setOutputs("30")
                        .setInputTypes(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                        .build();

        return conf;
    }

    public ComputationGraph init() {
        ComputationGraphConfiguration graph = this.computationGraph();
        ComputationGraph model = new ComputationGraph(graph);
        model.init();
        return model;
    }

}
