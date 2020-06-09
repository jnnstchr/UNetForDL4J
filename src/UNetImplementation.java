import lombok.AllArgsConstructor;
import lombok.Builder;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
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
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.File;
import java.util.Random;

@AllArgsConstructor
@Builder
public class UNetImplementation {

    protected static final Logger log = LoggerFactory.getLogger(UNetImplementation.class);
    private static final int seed = 123;
    private WeightInit weightInit = WeightInit.RELU;
    protected static String modelType = "UNet";
    protected static Random rng = new Random(seed);
    protected static int epochs = 1;
    protected static boolean save = false;
    private int numLabels;
    int batchSize = 1;

    double learningRate = 1e-4;

    public static void main(String[] args) throws Exception {

        int width = 512;
        int height = 512;
        int channels = 1;
        int outputNum = 2;

        File trainData = new File("/home/jstachera/dev/GSOC-2020/ISBI-DATASET/train");
        File testData = new File("/home/jstachera/dev/GSOC-2020/ISBI-DATASET/test");
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 1, 1, outputNum);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        for (int i = 1; i < 3; i++) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());
        }
    }
    }