import lombok.AllArgsConstructor;
import lombok.Builder;
import org.bytedeco.mkl.global.mkl_rt;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.File;
import java.util.Random;

@AllArgsConstructor
@Builder
public class UNetImplementation {
    private static final Logger log = LoggerFactory.getLogger(UNetImplementation.class);
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 100;
    private static int batchSize = 6;

    double learningRate = 1e-4;
    private static int width = 512;
    private static int height = 512;
    private static int channels = 1;
    public static final String dataPath = "/home/jstachera/dev/GSOC-2020/ISBI-DATASET";
    public static void main(String[] args) throws Exception {

        File trainData = new File(dataPath + "/train/images");
        File testData = new File(dataPath + "test");
        LabelGenerator labelMaker = new LabelGenerator(dataPath);

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);

        ImageRecordReader rr = new ImageRecordReader(height, width, channels, labelMaker);

        rr.initialize(train);
        rr.setListeners(new LogRecordListener());
        int labelIndex = 1;

        DataSetIterator dataTrainIter = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, labelIndex, true);
        DataSetIterator dataTestIter = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, labelIndex, true);


        NormalizerMinMaxScaler  scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fitLabel(true);
        scaler.fit(dataTrainIter);
        dataTrainIter.setPreProcessor(scaler);

        for (int i = 1; i < 3; i++) {
            DataSet ds = dataTrainIter.next();
            System.out.println(ds);
            System.out.println(dataTrainIter.getLabels());
        }
    }
    }