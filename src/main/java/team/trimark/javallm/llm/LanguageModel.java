package team.trimark.javallm.llm;

import team.trimark.javallm.nn.LayerNorm;
import team.trimark.javallm.nn.Matrices;
import team.trimark.javallm.nn.Parameter;
import team.trimark.javallm.nn.TransformerBlock;
import team.trimark.javallm.type.TokenMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A decoder-only transformer language model. Given a sequence of tokens it predicts, for every
 * position at once, a probability distribution over the token which should come next.
 */
public final class LanguageModel {
    /**
     * The vocabulary this model predicts over.
     */
    private final Vocabulary vocabulary;

    /**
     * The number of features carried per position.
     */
    private final int features;

    /**
     * The longest sequence this model can attend over.
     */
    private final int maxSequence;

    /**
     * Every trainable parameter of this model.
     */
    private final List<Parameter> parameters = new ArrayList<>();

    /**
     * The token embedding table, mapping each vocabulary entry to a vector.
     */
    private final Parameter tokenEmbedding;

    /**
     * The position embedding table, mapping each position to a vector.
     */
    private final Parameter positionEmbedding;

    /**
     * The stack of transformer blocks.
     */
    private final List<TransformerBlock> blocks = new ArrayList<>();

    /**
     * The normalization applied before the output projection.
     */
    private final LayerNorm finalNorm;

    /**
     * The projection from features to vocabulary scores.
     */
    private final Parameter outputWeights;

    /**
     * The bias of the output projection.
     */
    private final Parameter outputBias;

    /**
     * The identifiers of the most recent forward pass.
     */
    private int[] inputIds;

    /**
     * The normalized activations feeding the output projection on the most recent forward pass.
     */
    private TokenMatrix finalActivations;

    /**
     * Creates a new language model.
     * @param vocabulary The vocabulary to predict over
     * @param features The number of features carried per position
     * @param hiddenFeatures The width of each feed-forward hidden layer
     * @param layers The number of transformer blocks to stack
     * @param maxSequence The longest sequence this model can attend over
     * @param seed The seed used to initialize the parameters
     */
    public LanguageModel(Vocabulary vocabulary, int features, int hiddenFeatures, int layers, int maxSequence, long seed) {
        this.vocabulary = vocabulary;
        this.features = features;
        this.maxSequence = maxSequence;

        Random random = new Random(seed);

        this.tokenEmbedding = new Parameter("tok", Matrices.randomNormal(vocabulary.size(), features, 0.02, random));
        this.positionEmbedding = new Parameter("pos", Matrices.randomNormal(maxSequence, features, 0.02, random));

        parameters.add(tokenEmbedding);
        parameters.add(positionEmbedding);

        for (int i = 0; i < layers; i++) {
            blocks.add(new TransformerBlock("block" + i, features, hiddenFeatures, parameters, random));
        }

        this.finalNorm = new LayerNorm("lnf", features, parameters);
        this.outputWeights = new Parameter("out.w", Matrices.randomNormal(features, vocabulary.size(), 0.02, random));
        this.outputBias = new Parameter("out.b", new TokenMatrix(1, vocabulary.size()));

        parameters.add(outputWeights);
        parameters.add(outputBias);
    }

    /**
     * Returns every trainable parameter of this model.
     * @return The parameters
     */
    public List<Parameter> parameters() {
        return parameters;
    }

    /**
     * Returns the total number of scalar values across every parameter.
     * @return The number of scalar values
     */
    public int parameterCount() {
        int total = 0;

        for (Parameter p : parameters) {
            total += p.count();
        }

        return total;
    }

    /**
     * Returns the vocabulary this model predicts over.
     * @return The vocabulary
     */
    public Vocabulary vocabulary() {
        return vocabulary;
    }

    /**
     * Returns the longest sequence this model can attend over.
     * @return The maximum sequence length
     */
    public int maxSequence() {
        return maxSequence;
    }

    /**
     * Runs the model over a sequence of identifiers.
     * @param ids The identifiers to run over
     * @return The unnormalized scores, of shape {@code sequence * vocabulary}
     * @throws IllegalArgumentException When the sequence is empty or longer than the maximum
     */
    public TokenMatrix forward(int[] ids) {
        if (ids.length == 0 || ids.length > maxSequence) {
            throw new IllegalArgumentException("Sequence length must be between 1 and " + maxSequence + ", got " + ids.length);
        }

        inputIds = ids;

        TokenMatrix x = new TokenMatrix(ids.length, features);

        for (int t = 0; t < ids.length; t++) {
            for (int c = 0; c < features; c++) {
                x.set(t, c, tokenEmbedding.value.get(ids[t], c) + positionEmbedding.value.get(t, c));
            }
        }

        for (TransformerBlock block : blocks) {
            x = block.forward(x);
        }

        finalActivations = finalNorm.forward(x);

        return Matrices.addRowVector(Matrices.matmul(finalActivations, outputWeights.value), outputBias.value);
    }

    /**
     * Runs the model, computes the mean cross-entropy against the provided targets, and accumulates
     * the gradient of that loss into every parameter.
     * @param ids The input identifiers
     * @param targets The identifier which should follow each input position
     * @return The mean cross-entropy loss, in nats
     * @throws IllegalArgumentException When the inputs and targets differ in length
     */
    public float accumulateGradients(int[] ids, int[] targets) {
        if (ids.length != targets.length) {
            throw new IllegalArgumentException("Inputs and targets must be the same length.");
        }

        TokenMatrix logits = forward(ids);
        TokenMatrix probabilities = Matrices.softmaxRows(logits);

        int t = ids.length;
        float loss = 0f;

        TokenMatrix dLogits = new TokenMatrix(t, vocabulary.size());

        for (int i = 0; i < t; i++) {
            loss -= (float) Math.log(Math.max(probabilities.get(i, targets[i]), 1e-12f));

            for (int j = 0; j < vocabulary.size(); j++) {
                float target = j == targets[i] ? 1f : 0f;
                dLogits.set(i, j, (probabilities.get(i, j) - target) / t);
            }
        }

        backward(dLogits);
        return loss / t;
    }

    /**
     * Propagates the gradient of the loss with respect to the scores back through the whole model.
     * @param dLogits The gradient with respect to the output scores
     */
    private void backward(TokenMatrix dLogits) {
        Matrices.addInPlace(outputWeights.gradient, Matrices.matmulTN(finalActivations, dLogits));
        Matrices.addInPlace(outputBias.gradient, Matrices.sumRows(dLogits));

        TokenMatrix dx = finalNorm.backward(Matrices.matmulNT(dLogits, outputWeights.value));

        for (int i = blocks.size() - 1; i >= 0; i--) {
            dx = blocks.get(i).backward(dx);
        }

        for (int t = 0; t < inputIds.length; t++) {
            for (int c = 0; c < features; c++) {
                float g = dx.get(t, c);

                tokenEmbedding.gradient.set(inputIds[t], c, tokenEmbedding.gradient.get(inputIds[t], c) + g);
                positionEmbedding.gradient.set(t, c, positionEmbedding.gradient.get(t, c) + g);
            }
        }
    }

    /**
     * Continues the provided prompt by repeatedly sampling the next token from the model.
     * @param prompt The text to continue
     * @param newTokens The number of tokens to generate
     * @param temperature The sampling temperature; lower is more deterministic
     * @param topK The number of highest scoring candidates to sample from
     * @param random The source of randomness
     * @return The prompt followed by the generated continuation
     */
    public String generate(String prompt, int newTokens, float temperature, int topK, Random random) {
        int[] context = vocabulary.encode(prompt);

        if (context.length == 0) {
            throw new IllegalArgumentException("Prompt contains no known tokens.");
        }

        int[] produced = new int[context.length + newTokens];
        System.arraycopy(context, 0, produced, 0, context.length);
        int length = context.length;

        for (int step = 0; step < newTokens; step++) {
            int windowStart = Math.max(0, length - maxSequence);
            int windowLength = length - windowStart;

            int[] window = new int[windowLength];
            System.arraycopy(produced, windowStart, window, 0, windowLength);

            TokenMatrix logits = forward(window);
            produced[length++] = sampleFrom(logits, windowLength - 1, temperature, topK, random);
        }

        int[] result = new int[length];
        System.arraycopy(produced, 0, result, 0, length);
        return vocabulary.decode(result);
    }

    /**
     * Samples one identifier from the scores of a single position.
     * @param logits The score matrix
     * @param row The position to sample from
     * @param temperature The sampling temperature
     * @param topK The number of highest scoring candidates to consider
     * @param random The source of randomness
     * @return The sampled identifier
     */
    private int sampleFrom(TokenMatrix logits, int row, float temperature, int topK, Random random) {
        int size = vocabulary.size();
        int k = Math.min(Math.max(topK, 1), size);

        int[] best = new int[k];
        float[] bestScores = new float[k];
        java.util.Arrays.fill(bestScores, Float.NEGATIVE_INFINITY);

        for (int j = 0; j < size; j++) {
            float score = logits.get(row, j) / Math.max(temperature, 1e-6f);

            for (int slot = 0; slot < k; slot++) {
                if (score > bestScores[slot]) {
                    for (int shift = k - 1; shift > slot; shift--) {
                        bestScores[shift] = bestScores[shift - 1];
                        best[shift] = best[shift - 1];
                    }

                    bestScores[slot] = score;
                    best[slot] = j;
                    break;
                }
            }
        }

        float sum = 0f;
        float max = bestScores[0];

        for (int slot = 0; slot < k; slot++) {
            bestScores[slot] = (float) Math.exp(bestScores[slot] - max);
            sum += bestScores[slot];
        }

        float target = random.nextFloat() * sum;
        float running = 0f;

        for (int slot = 0; slot < k; slot++) {
            running += bestScores[slot];

            if (running >= target) {
                return best[slot];
            }
        }

        return best[k - 1];
    }
}
