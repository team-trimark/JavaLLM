package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

import java.util.List;
import java.util.Random;

/**
 * One transformer block: normalize, attend, add back; then normalize, transform, add back. The two
 * additions form the residual stream, which lets gradients reach the earliest layers unattenuated.
 */
public final class TransformerBlock {
    /**
     * The normalization applied before attention.
     */
    private final LayerNorm attentionNorm;

    /**
     * The attention head.
     */
    private final CausalSelfAttention attention;

    /**
     * The normalization applied before the feed-forward network.
     */
    private final LayerNorm feedForwardNorm;

    /**
     * The feed-forward network.
     */
    private final FeedForward feedForward;

    /**
     * Creates a new transformer block.
     * @param name The name prefix used for the parameters
     * @param features The number of features per position
     * @param hiddenFeatures The width of the feed-forward hidden layer
     * @param parameters The registry to add the new parameters to
     * @param random The source of randomness for initialization
     */
    public TransformerBlock(String name, int features, int hiddenFeatures, List<Parameter> parameters, Random random) {
        this.attentionNorm = new LayerNorm(name + ".ln1", features, parameters);
        this.attention = new CausalSelfAttention(name + ".attn", features, parameters, random);
        this.feedForwardNorm = new LayerNorm(name + ".ln2", features, parameters);
        this.feedForward = new FeedForward(name + ".ffn", features, hiddenFeatures, parameters, random);
    }

    /**
     * Runs this block over the provided activations.
     * @param x The activations, of shape {@code sequence * features}
     * @return The output, of the same shape
     */
    public TokenMatrix forward(TokenMatrix x) {
        TokenMatrix afterAttention = Matrices.add(x, attention.forward(attentionNorm.forward(x)));
        return Matrices.add(afterAttention, feedForward.forward(feedForwardNorm.forward(afterAttention)));
    }

    /**
     * Propagates the gradient back through the most recent forward pass.
     * @param dOut The gradient with respect to the output of this block
     * @return The gradient with respect to the input of this block
     */
    public TokenMatrix backward(TokenMatrix dOut) {
        TokenMatrix dAfterAttention = Matrices.add(dOut, feedForwardNorm.backward(feedForward.backward(dOut)));
        return Matrices.add(dAfterAttention, attentionNorm.backward(attention.backward(dAfterAttention)));
    }
}
