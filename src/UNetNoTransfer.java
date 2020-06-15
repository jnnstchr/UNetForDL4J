//import org.datavec.api.split.FileSplit;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.recordreader.ImageRecordReader;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.*;
//import org.deeplearning4j.nn.conf.graph.MergeVertex;
//import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.*;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.api.buffer.DataType;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.learning.config.AdaDelta;
//import org.nd4j.linalg.learning.config.IUpdater;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.Random;
//
//public class UNetNoTransfer {
//    private static final int seed = 1234;
//    private WeightInit weightInit = WeightInit.RELU;
//    protected static Random rng = new Random(seed);
//    protected static int epochs = 1;
//    private static int batchSize = 1;
//    private int[] inputShape = new int[] {3, 512, 512};
//
//    private static int width = 512;
//    private static int height = 512;
//    private static int channels = 3;
//    public static final String dataPath = "/home/jstachera/Documents/data";
//
//    public static void main(String[] args) throws IOException {
//
//        File trainData = new File(dataPath + "/train/image");
//        File testData = new File(dataPath + "/test/image");
//        LabelGenerator labelMakerTrain = new LabelGenerator(dataPath + "/train");
//        LabelGenerator labelMakerTest = new LabelGenerator(dataPath + "/test");
//
//        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
//        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);
//
//
//        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMakerTrain);
//        rrTrain.initialize(train, null);
//
//        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMakerTest);
//        rrTest.initialize(test, null);
//
//        int labelIndex = 1;
//
//        DataSetIterator dataTrainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, labelIndex, true);
//        DataSetIterator dataTestIter = new RecordReaderDataSetIterator(rrTest, 1, labelIndex, labelIndex, true);
//        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
//        scaler.fitLabel(true);
//        scaler.fit(dataTrainIter);
//        dataTrainIter.setPreProcessor(scaler);
//        scaler.fit(dataTestIter);
//        dataTestIter.setPreProcessor(scaler);
//
//        UNetNoTransfer un=new UNetNoTransfer();
//        ComputationGraphConfiguration.GraphBuilder graph = un.unetBuilder();
//
//        graph.addInputs("input").setInputTypes(InputType.convolutional3D(Convolution3D.DataFormat.NCDHW,un.inputShape[0],un.inputShape[1],un.inputShape[2]));
//
//        ComputationGraphConfiguration conf = graph.build();
//        ComputationGraph cg = new ComputationGraph(conf);
//        cg.init();
//        System.out.println(cg.summary());
//    }
//
//
//    private CacheMode cacheMode = CacheMode.NONE;
//    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
//    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
//    private IUpdater updater = new AdaDelta();
//
//
//    public  ComputationGraphConfiguration.GraphBuilder unetBuilder() {
//
//        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(updater)
//                .weightInit(weightInit)
//                .l2(5e-5)
//                .miniBatch(true)
//                .cacheMode(cacheMode)
//                .trainingWorkspaceMode(workspaceMode)
//                .inferenceWorkspaceMode(workspaceMode)
//                .graphBuilder();
//
//
//        graph
//                .addLayer("conv1-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(32).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "input")
//                .addLayer("conv1-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(64).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv1-1")
//                .addLayer("pool1", new Subsampling3DLayer.Builder(Subsampling3DLayer.PoolingType.MAX).kernelSize(2,2,2)
//                        .build(), "conv1-2")
//
//                .addLayer("conv2-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(64).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "pool1")
//                .addLayer("conv2-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(128).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv2-1")
//                .addLayer("pool2", new Subsampling3DLayer.Builder(Subsampling3DLayer.PoolingType.MAX).kernelSize(2,2,2)
//                        .build(), "conv2-2")
//
//                .addLayer("conv3-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(128).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "pool2")
//                .addLayer("conv3-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(256).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv3-1")
//                .addLayer("drop3", new DropoutLayer.Builder(0.5).build(), "conv3-2")
//                .addLayer("pool3", new Subsampling3DLayer.Builder(Subsampling3DLayer.PoolingType.MAX).kernelSize(2,2,2)
//                        .build(), "drop3")
//
//                .addLayer("conv4-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(256).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "pool3")
//                .addLayer("conv4-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(512).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv4-1")
//                .addLayer("drop4", new DropoutLayer.Builder(0.5).build(), "conv4-2")
//                // up5
//                .addLayer("up5-1", new Upsampling3D.Builder(2).build(), "drop4")
//                .addLayer("up5-2", new Convolution3D.Builder(2,2,2).stride(1,1,1).nOut(512).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "up5-1")
//                .addVertex("merge5", new MergeVertex(), "drop3", "up5-2")
//                .addLayer("conv5-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(256).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "merge5")
//                .addLayer("conv5-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(256).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv5-1")
//
//                // up6
//                .addLayer("up6-1", new Upsampling3D.Builder(2).build(), "conv5-2")
//                .addLayer("up6-2", new Convolution3D.Builder(2,2,2).stride(1,1,1).nOut(256).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "up6-1")
//                .addVertex("merge6", new MergeVertex(), "conv2-2", "up6-2")
//                .addLayer("conv6-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(128).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "merge6")
//                .addLayer("conv6-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(128).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv6-1")
//
//                // up7
//                .addLayer("up7-1", new Upsampling3D.Builder(2).build(), "conv6-2")
//                .addLayer("up7-2", new Convolution3D.Builder(2,2,2).stride(1,1,1).nOut(128).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "up7-1")
//                .addVertex("merge7", new MergeVertex(), "conv1-2", "up7-2")
//                .addLayer("conv7-1", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(64).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "merge7")
//                .addLayer("conv7-2", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(64).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv7-1")
//                .addLayer("conv7-3", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(2)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.RELU).build(), "conv7-2")
//
//                .addLayer("conv8", new Convolution3D.Builder(3,3,3).stride(1,1,1).nOut(1).dataFormat(Convolution3D.DataFormat.NCDHW)
//                        .convolutionMode(ConvolutionMode.Same).cudnnAlgoMode(cudnnAlgoMode)
//                        .activation(Activation.IDENTITY).build(), "conv7-3")
//                .addLayer("output", new Cnn3DLossLayer.Builder(Convolution3D.DataFormat.NCDHW).lossFunction(LossFunctions.LossFunction.XENT)
//                        .activation(Activation.SIGMOID).build(), "conv8")
//
//                .setOutputs("output");
//
//        return graph;
//    }
//}