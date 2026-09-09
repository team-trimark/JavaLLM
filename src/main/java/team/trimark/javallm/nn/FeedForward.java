package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

import java.util.List;
import java.util.Random;

/**
 * A position-wise feed-forward network. Each position is expanded into a wider hidden layer, passed
 * through a rectified linear unit, and projected back down. Unlike attention, this mixes features
 * within a position rather than across positions.
 */
public final class FeedForward {
    /**
     * The expanding projection.
     */
    private final Parameter inputWeights;

    /**
     * The bias of the expanding projection.
     */
    private final Parameter inputBias;

    /**
     * The contracting projection.
     */
    private final Parameter outputWeights;

    /**
     * The bias of the contracting projection.
     */
    private final Parameter outputBias;

    /**
     * The input of the most recent forward pass.
     */
    private TokenMatrix input;

    /**
     * The hidden activations of the most recent forward pass, after the rectifier.
     */
    private TokenMatrix hidden;

    /**
     * Creates a new feed-forward network.
     * @param name The name prefix used for the parameters
     * @param features The number of features per position
     * @param hiddenFeatures The width of the hidden layer
     * @param parameters The registry to add the new parameters to
     * @param random The source of randomness for initialization
     */
    public FeedForward(String name, int features, int hiddenFeatures, List<Parameter> parameters, Random random) {
        this.inputWeights = new Parameter(name + ".w1", Matrices.randomNormal(features, hiddenFeatures, 0.02, random));
        this.inputBias = new Parameter(name + ".b1", new TokenMatrix(1, hiddenFeatures));
        this.outputWeights = new Parameter(name + ".w2", Matrices.randomNormal(hiddenFeatures, features, 0.02, random));
        this.outputBias = new Parameter(name + ".b2", new TokenMatrix(1, features));

        parameters.add(inputWeights);
        parameters.add(inputBias);
        parameters.add(outputWeights);
        parameters.add(outputBias);
    }

    /**
     * Runs the network over the provided activations.
     * @param x The activations, of shape {@code sequence * features}
     * @return The output, of the same shape
     */
    public TokenMatrix forward(TokenMatrix x) {
        input = x;

        TokenMatrix pre = Matrices.addRowVector(Matrices.matmul(x, inputWeights.value), inputBias.value);
        hidden = new TokenMatrix(pre.rows(), pre.columns());

        for (int r = 0; r < pre.rows(); r++) {
            for (int c = 0; c < pre.columns(); c++) {
                hidden.set(r, c, Math.max(0f, pre.get(r, c)));
            }
        }

        return Matrices.addRowVector(Matrices.matmul(hidden, outputWeights.value), outputBias.value);
    }

    /**
     * Propagates the gradient back through the most recent forward pass, accumulating the gradients
     * of both projections and both biases.
     * @param dOut The gradient with respect to the output of this layer
     * @return The gradient with respect to the input of this layer
     */
    public TokenMatrix backward(TokenMatrix dOut) {
        Matrices.addInPlace(outputWeights.gradient, Matrices.matmulTN(hidden, dOut));
        Matrices.addInPlace(outputBias.gradient, Matrices.sumRows(dOut));

        TokenMatrix dHidden = Matrices.matmulNT(dOut, outputWeights.value);

        for (int r = 0; r < dHidden.rows(); r++) {
            for (int c = 0; c < dHidden.columns(); c++) {
                if (hidden.get(r, c) <= 0f) {
                    dHidden.set(r, c, 0f);
                }
            }
        }

        Matrices.addInPlace(inputWeights.gradient, Matrices.matmulTN(input, dHidden));
        Matrices.addInPlace(inputBias.gradient, Matrices.sumRows(dHidden));

        return Matrices.matmulNT(dHidden, inputWeights.value);
    }
}
