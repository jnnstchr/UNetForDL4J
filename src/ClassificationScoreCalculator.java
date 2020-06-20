import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<Model, Evaluation> {

    protected final Evaluation.Metric metric;

    public ClassificationScoreCalculator(Evaluation.Metric metric, DataSetIterator iterator) {
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected Evaluation newEval() {
        return new Evaluation();
    }

    @Override
    protected double finalScore(Evaluation e) {
        return e.scoreForMetric(metric);
    }

    @Override
    public boolean minimizeScore() {
        return false;
    }
}