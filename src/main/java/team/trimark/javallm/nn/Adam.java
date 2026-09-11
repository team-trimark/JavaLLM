package team.trimark.javallm.nn;

import team.trimark.javallm.type.DenseTokenMatrix;
import team.trimark.javallm.type.TokenMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The Adam optimizer, which adapts a per-value learning rate from running estimates of the first and
 * second moments of each gradient.
 */
public final class Adam {
    /**
     * The parameters being optimized.
     */
    private final List<Parameter> parameters;

    /**
     * The running first moment estimate of each parameter.
     */
    private final List<TokenMatrix> firstMoments = new ArrayList<>();

    /**
     * The running second moment estimate of each parameter.
     */
    private final List<TokenMatrix> secondMoments = new ArrayList<>();

    /**
     * The learning rate.
     */
    private float learningRate;

    /**
     * The decay rate of the first moment estimate.
     */
    private final float beta1;

    /**
     * The decay rate of the second moment estimate.
     */
    private final float beta2;

    /**
     * The constant added to the denominator for numerical stability.
     */
    private final float epsilon;

    /**
     * The number of steps taken so far, used for bias correction.
     */
    private int step = 0;

    /**
     * Creates a new optimizer over the provided parameters.
     * @param parameters The parameters to optimize
     * @param learningRate The learning rate
     * @throws NullPointerException When the parameter list is {@code null}
     */
    public Adam(List<Parameter> parameters, float learningRate) {
        this.parameters = Objects.requireNonNull(parameters, "Parameters cannot be null.");
        this.learningRate = learningRate;
        this.beta1 = 0.9f;
        this.beta2 = 0.999f;
        this.epsilon = 1e-8f;

        for (Parameter p : parameters) {
            firstMoments.add(TokenMatrix.of(p.value.rows(), p.value.columns()));
            secondMoments.add(TokenMatrix.of(p.value.rows(), p.value.columns()));
        }
    }

    /**
     * Returns the learning rate every step is currently scaled by.
     * @return The learning rate
     */
    public float learningRate() {
        return learningRate;
    }

    /**
     * Sets the learning rate applied by subsequent steps. A long run benefits from
     * decaying this over time: a large rate early covers ground quickly, while a smaller
     * rate late lets the parameters settle instead of bouncing around the minimum. The
     * moment estimates are untouched, so a change takes effect smoothly.
     * @param learningRate The new learning rate
     * @throws IllegalArgumentException When the learning rate is not positive
     */
    public void learningRate(float learningRate) {
        if (!(learningRate > 0f)) {
            throw new IllegalArgumentException("Learning rate must be positive, got " + learningRate + ".");
        }

        this.learningRate = learningRate;
    }

    /**
     * Applies one optimization step to every parameter using its currently accumulated gradient.
     */
    public void step() {
        step++;

        float correction1 = 1f - (float) Math.pow(beta1, step);
        float correction2 = 1f - (float) Math.pow(beta2, step);

        for (int i = 0; i < parameters.size(); i++) {
            Parameter p = parameters.get(i);
            TokenMatrix m = firstMoments.get(i);
            TokenMatrix v = secondMoments.get(i);

            float[] pv = raw(p.value);
            float[] pg = raw(p.gradient);
            float[] mv = raw(m);
            float[] vv = raw(v);

            if (pv != null && pg != null && mv != null && vv != null) {
                // Every value is updated independently of every other, so the order the values are
                // walked in cannot change the result. Walking the backing arrays straight through
                // reads each of the four in one contiguous sweep, where the row-column form below
                // strides across memory on every step and misses cache on all four at once.
                for (int j = 0; j < pv.length; j++) {
                    float g = pg[j];

                    float mNew = beta1 * mv[j] + (1f - beta1) * g;
                    float vNew = beta2 * vv[j] + (1f - beta2) * g * g;

                    mv[j] = mNew;
                    vv[j] = vNew;

                    float mHat = mNew / correction1;
                    float vHat = vNew / correction2;

                    pv[j] = pv[j] - learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
                }

                continue;
            }

            for (int r = 0; r < p.value.rows(); r++) {
                for (int c = 0; c < p.value.columns(); c++) {
                    float g = p.gradient.get(r, c);

                    float mNew = beta1 * m.get(r, c) + (1f - beta1) * g;
                    float vNew = beta2 * v.get(r, c) + (1f - beta2) * g * g;

                    m.set(r, c, mNew);
                    v.set(r, c, vNew);

                    float mHat = mNew / correction1;
                    float vHat = vNew / correction2;

                    p.value.set(r, c, p.value.get(r, c) - learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon));
                }
            }
        }
    }

    /**
     * Returns the writable backing array of a matrix, when it has one.
     * @param m The matrix to unwrap
     * @return The backing array, or {@code null} when the matrix does not expose one
     */
    private static float[] raw(TokenMatrix m) {
        return m instanceof DenseTokenMatrix dense ? dense.columnMajorValues() : null;
    }

    /**
     * Resets the accumulated gradient of every parameter to zero.
     */
    public void zeroGradients() {
        for (Parameter p : parameters) {
            p.zeroGradient();
        }
    }
}
