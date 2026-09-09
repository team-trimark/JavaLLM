package team.trimark.javallm.nn;

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
    private final float learningRate;

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
            firstMoments.add(new TokenMatrix(p.value.rows(), p.value.columns()));
            secondMoments.add(new TokenMatrix(p.value.rows(), p.value.columns()));
        }
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
     * Resets the accumulated gradient of every parameter to zero.
     */
    public void zeroGradients() {
        for (Parameter p : parameters) {
            p.zeroGradient();
        }
    }
}
