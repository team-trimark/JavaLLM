package team.trimark.javallm.llm;

import team.trimark.javallm.nn.Parameter;
import team.trimark.javallm.type.TokenMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Verifies the hand written backward pass against numerical differentiation.
 * <p>
 * Rather than perturbing one value at a time, which drowns in {@code float} rounding noise, this
 * takes a directional derivative: every value of a parameter is nudged along a random unit vector
 * and the measured slope is compared against the same projection of the analytic gradient. That
 * aggregates the signal of the whole parameter while keeping the perturbation small.
 */
public final class GradientCheck {
    /**
     * The size of the nudge applied along each direction.
     */
    private static final double EPSILON = 5e-4;

    /**
     * The largest relative error accepted before a parameter is reported as failing. The rectified
     * linear units make the loss only piecewise differentiable, so a central difference that steps
     * across a kink disagrees with the true derivative no matter how correct the derivative is. The
     * tolerance is loose enough to absorb that.
     */
    private static final double TOLERANCE = 0.10;

    /**
     * The magnitude below which a directional derivative is considered indistinguishable from noise.
     * The loss is accumulated in {@code float}, so its resolution near a value of one is roughly
     * {@code 1e-7}; dividing that by the step size bounds how small a slope can still be measured.
     */
    private static final double NOISE_FLOOR = 1e-3;

    /**
     * Private constructor to prevent instantiation.
     */
    private GradientCheck() {}

    /**
     * Runs the gradient check over a small model and reports the result of every parameter.
     * @param args The command line arguments, which are ignored
     */
    public static void main(String[] args) {
        Vocabulary vocabulary = new Vocabulary("abcdefgh ijkl");
        LanguageModel model = new LanguageModel(vocabulary, 16, 32, 2, 8, 7L);

        int[] ids = vocabulary.encode("abcd efg");
        int[] targets = vocabulary.encode("bcd efgh");

        model.parameters().forEach(Parameter::zeroGradient);
        model.accumulateGradients(ids, targets);

        List<float[]> analyticGradients = new ArrayList<>();

        for (Parameter p : model.parameters()) {
            analyticGradients.add(flatten(p.gradient));
        }

        Random random = new Random(5L);
        int failed = 0;
        int skipped = 0;

        System.out.println("Directional gradient check, epsilon=" + EPSILON);

        for (int i = 0; i < model.parameters().size(); i++) {
            Parameter p = model.parameters().get(i);
            float[] direction = randomUnitVector(p.count(), random);
            float[] analyticGradient = analyticGradients.get(i);

            double analytic = 0;

            for (int j = 0; j < analyticGradient.length; j++) {
                analytic += analyticGradient[j] * (double) direction[j];
            }

            shift(p, direction, EPSILON);
            double lossUp = lossWithoutDisturbingGradients(model, ids, targets);

            shift(p, direction, -2 * EPSILON);
            double lossDown = lossWithoutDisturbingGradients(model, ids, targets);

            shift(p, direction, EPSILON);

            double numerical = (lossUp - lossDown) / (2 * EPSILON);
            double relative = Math.abs(analytic - numerical) / Math.max(1e-9, Math.abs(analytic) + Math.abs(numerical));

            String verdict;

            if (Math.abs(analytic) < NOISE_FLOOR && Math.abs(numerical) < NOISE_FLOOR) {
                verdict = "skipped (below noise floor)";
                skipped++;
            } else if (relative > TOLERANCE) {
                verdict = "FAILED";
                failed++;
            } else {
                verdict = "ok";
            }

            System.out.printf("  %-18s analytic=%+.6f numerical=%+.6f relative=%.4f  %s%n",
                    p.name, analytic, numerical, relative, verdict);
        }

        System.out.println();
        System.out.println("Parameters: " + model.parameters().size() + ", skipped: " + skipped + ", failed: " + failed);
        System.out.println(failed == 0 ? "GRADIENT CHECK PASSED" : "GRADIENT CHECK FAILED");
    }

    /**
     * Builds a random vector of unit length.
     * @param length The number of values
     * @param random The source of randomness
     * @return The unit vector
     */
    private static float[] randomUnitVector(int length, Random random) {
        float[] direction = new float[length];
        double norm = 0;

        for (int i = 0; i < length; i++) {
            direction[i] = (float) random.nextGaussian();
            norm += direction[i] * (double) direction[i];
        }

        norm = Math.sqrt(norm);

        for (int i = 0; i < length; i++) {
            direction[i] /= (float) norm;
        }

        return direction;
    }

    /**
     * Nudges every value of a parameter along the provided direction.
     * @param parameter The parameter to nudge
     * @param direction The direction to move along
     * @param scale The distance to move
     */
    private static void shift(Parameter parameter, float[] direction, double scale) {
        int i = 0;

        for (int r = 0; r < parameter.value.rows(); r++) {
            for (int c = 0; c < parameter.value.columns(); c++) {
                parameter.value.set(r, c, (float) (parameter.value.get(r, c) + scale * direction[i++]));
            }
        }
    }

    /**
     * Flattens a matrix into an array in row major order.
     * @param matrix The matrix to flatten
     * @return The flattened values
     */
    private static float[] flatten(TokenMatrix matrix) {
        float[] flat = new float[matrix.rows() * matrix.columns()];
        int i = 0;

        for (int r = 0; r < matrix.rows(); r++) {
            for (int c = 0; c < matrix.columns(); c++) {
                flat[i++] = matrix.get(r, c);
            }
        }

        return flat;
    }

    /**
     * Restores a matrix from an array in row major order.
     * @param matrix The matrix to restore into
     * @param flat The values to restore
     */
    private static void unflatten(TokenMatrix matrix, float[] flat) {
        int i = 0;

        for (int r = 0; r < matrix.rows(); r++) {
            for (int c = 0; c < matrix.columns(); c++) {
                matrix.set(r, c, flat[i++]);
            }
        }
    }

    /**
     * Computes the loss of the model, leaving the accumulated gradients exactly as they were.
     * @param model The model to evaluate
     * @param ids The input identifiers
     * @param targets The target identifiers
     * @return The mean cross-entropy loss
     */
    private static double lossWithoutDisturbingGradients(LanguageModel model, int[] ids, int[] targets) {
        List<float[]> saved = new ArrayList<>();

        for (Parameter p : model.parameters()) {
            saved.add(flatten(p.gradient));
        }

        double value = model.accumulateGradients(ids, targets);
        int i = 0;

        for (Parameter p : model.parameters()) {
            unflatten(p.gradient, saved.get(i++));
        }

        return value;
    }
}
