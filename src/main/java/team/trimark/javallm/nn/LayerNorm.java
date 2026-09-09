package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

import java.util.List;

/**
 * Layer normalization. Each row is independently rescaled to zero mean and unit variance, then
 * shifted and scaled by learned per-feature parameters. This keeps activations in a stable range as
 * they flow through the residual stream.
 */
public final class LayerNorm {
    /**
     * The constant added to the variance for numerical stability.
     */
    private static final float EPSILON = 1e-5f;

    /**
     * The learned per-feature scale.
     */
    private final Parameter gamma;

    /**
     * The learned per-feature shift.
     */
    private final Parameter beta;

    /**
     * The normalized activations of the most recent forward pass.
     */
    private TokenMatrix normalized;

    /**
     * The reciprocal standard deviation of each row of the most recent forward pass.
     */
    private float[] inverseDeviations;

    /**
     * Creates a new layer normalization over the provided number of features.
     * @param name The name prefix used for the parameters
     * @param features The number of features per row
     * @param parameters The registry to add the new parameters to
     */
    public LayerNorm(String name, int features, List<Parameter> parameters) {
        this.gamma = new Parameter(name + ".gamma", Matrices.filled(1, features, 1f));
        this.beta = new Parameter(name + ".beta", new TokenMatrix(1, features));

        parameters.add(gamma);
        parameters.add(beta);
    }

    /**
     * Normalizes every row of the provided activations.
     * @param x The activations, of shape {@code sequence * features}
     * @return The normalized activations, of the same shape
     */
    public TokenMatrix forward(TokenMatrix x) {
        int t = x.rows();
        int d = x.columns();

        normalized = new TokenMatrix(t, d);
        inverseDeviations = new float[t];

        TokenMatrix out = new TokenMatrix(t, d);

        for (int r = 0; r < t; r++) {
            float mean = 0f;

            for (int c = 0; c < d; c++) {
                mean += x.get(r, c);
            }

            mean /= d;

            float variance = 0f;

            for (int c = 0; c < d; c++) {
                float diff = x.get(r, c) - mean;
                variance += diff * diff;
            }

            variance /= d;

            float inverseDeviation = 1f / (float) Math.sqrt(variance + EPSILON);
            inverseDeviations[r] = inverseDeviation;

            for (int c = 0; c < d; c++) {
                float hat = (x.get(r, c) - mean) * inverseDeviation;
                normalized.set(r, c, hat);
                out.set(r, c, hat * gamma.value.get(0, c) + beta.value.get(0, c));
            }
        }

        return out;
    }

    /**
     * Propagates the gradient back through the most recent forward pass, accumulating the gradients
     * of the scale and shift parameters.
     * @param dOut The gradient with respect to the output of this layer
     * @return The gradient with respect to the input of this layer
     */
    public TokenMatrix backward(TokenMatrix dOut) {
        int t = dOut.rows();
        int d = dOut.columns();

        TokenMatrix dx = new TokenMatrix(t, d);

        for (int r = 0; r < t; r++) {
            float meanDHat = 0f;
            float meanDHatTimesHat = 0f;

            for (int c = 0; c < d; c++) {
                float hat = normalized.get(r, c);
                float dOutValue = dOut.get(r, c);

                gamma.gradient.set(0, c, gamma.gradient.get(0, c) + dOutValue * hat);
                beta.gradient.set(0, c, beta.gradient.get(0, c) + dOutValue);

                float dHat = dOutValue * gamma.value.get(0, c);
                meanDHat += dHat;
                meanDHatTimesHat += dHat * hat;
            }

            meanDHat /= d;
            meanDHatTimesHat /= d;

            for (int c = 0; c < d; c++) {
                float hat = normalized.get(r, c);
                float dHat = dOut.get(r, c) * gamma.value.get(0, c);

                dx.set(r, c, inverseDeviations[r] * (dHat - meanDHat - hat * meanDHatTimesHat));
            }
        }

        return dx;
    }
}
