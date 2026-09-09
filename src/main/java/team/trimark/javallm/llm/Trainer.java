package team.trimark.javallm.llm;

import team.trimark.javallm.nn.Adam;
import team.trimark.javallm.nn.Parameter;

import java.util.Objects;
import java.util.Random;

/**
 * Trains a {@link LanguageModel} by repeatedly showing it windows of a corpus and asking it to
 * predict the next token at every position.
 */
public final class Trainer {
    /**
     * The model being trained.
     */
    private final LanguageModel model;

    /**
     * The optimizer which applies the accumulated gradients.
     */
    private final Adam optimizer;

    /**
     * The corpus, already encoded into vocabulary identifiers.
     */
    private final int[] corpus;

    /**
     * The source of randomness used to choose training windows.
     */
    private final Random random;

    /**
     * The largest magnitude any single gradient value is allowed to reach.
     */
    private final float gradientClip;

    /**
     * Creates a new trainer.
     * @param model The model to train
     * @param optimizer The optimizer to step with
     * @param corpus The training text
     * @param learningRate The learning rate, used only for reporting
     * @param gradientClip The largest magnitude any gradient value may take
     * @param seed The seed used to choose training windows
     * @throws NullPointerException When any argument is {@code null}
     * @throws IllegalArgumentException When the corpus is too short to form a single window
     */
    public Trainer(LanguageModel model, Adam optimizer, String corpus, float learningRate, float gradientClip, long seed) {
        this.model = Objects.requireNonNull(model, "Model cannot be null.");
        this.optimizer = Objects.requireNonNull(optimizer, "Optimizer cannot be null.");
        this.corpus = model.vocabulary().encode(Objects.requireNonNull(corpus, "Corpus cannot be null."));
        this.random = new Random(seed);
        this.gradientClip = gradientClip;

        if (this.corpus.length < 2) {
            throw new IllegalArgumentException("The corpus must contain at least two tokens.");
        }
    }

    /**
     * Runs one optimization step over a batch of randomly chosen windows.
     * @param batchSize The number of windows to average over
     * @param sequenceLength The number of tokens in each window
     * @return The mean loss over the batch, in nats per token
     * @throws IllegalArgumentException When the batch size or sequence length is not positive
     */
    public float step(int batchSize, int sequenceLength) {
        if (batchSize <= 0 || sequenceLength <= 0) {
            throw new IllegalArgumentException("Batch size and sequence length must be positive.");
        }

        int length = Math.min(sequenceLength, Math.min(model.maxSequence(), corpus.length - 1));

        optimizer.zeroGradients();
        float total = 0f;

        for (int i = 0; i < batchSize; i++) {
            int start = random.nextInt(corpus.length - length);

            int[] ids = new int[length];
            int[] targets = new int[length];

            System.arraycopy(corpus, start, ids, 0, length);
            System.arraycopy(corpus, start + 1, targets, 0, length);

            total += model.accumulateGradients(ids, targets);
        }

        scaleAndClipGradients(1f / batchSize);
        optimizer.step();

        return total / batchSize;
    }

    /**
     * Divides every accumulated gradient by the batch size and limits its magnitude. Without the
     * limit a single surprising window can take a step large enough to undo all prior progress.
     * @param scale The factor to multiply every gradient by
     */
    private void scaleAndClipGradients(float scale) {
        for (Parameter p : model.parameters()) {
            for (int r = 0; r < p.gradient.rows(); r++) {
                for (int c = 0; c < p.gradient.columns(); c++) {
                    float scaled = p.gradient.get(r, c) * scale;
                    p.gradient.set(r, c, Math.max(-gradientClip, Math.min(gradientClip, scaled)));
                }
            }
        }
    }
}
