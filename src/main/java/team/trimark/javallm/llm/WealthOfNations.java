package team.trimark.javallm.llm;

import team.trimark.javallm.nn.Adam;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Trains a language model on the full text of Adam Smith's <i>An Inquiry into the Nature
 * and Causes of the Wealth of Nations</i>.
 * <p>
 * Unlike {@link Demo}, which trains on ten repeated sentences in a few seconds, this runs
 * for hours over 2.2 million characters of real prose. Two things follow from that. The
 * run is checkpointed, so it can be interrupted and resumed rather than restarted. And the
 * learning rate is scheduled rather than fixed, because over thousands of steps the rate
 * that makes early progress is too large to settle at the end.
 * <p>
 * Usage: {@code WealthOfNations [--steps n] [--resume] [--checkpoint path] [--corpus path]}
 */
public final class WealthOfNations {
    /**
     * The number of features carried per position.
     */
    private static final int FEATURES = 384;

    /**
     * The width of each feed-forward hidden layer.
     */
    private static final int HIDDEN_FEATURES = 4 * FEATURES;

    /**
     * The number of transformer blocks stacked.
     */
    private static final int LAYERS = 8;

    /**
     * The longest sequence the model can attend over.
     */
    private static final int MAX_SEQUENCE = 512;

    /**
     * The number of windows averaged into each step. Every window contributes an
     * independent estimate of the gradient, so a wider batch trades fewer, slower steps
     * for a cleaner direction on each one. Too narrow and the noise between windows
     * swamps the signal, and the loss stalls rather than descends.
     */
    private static final int BATCH_SIZE = 12;

    /**
     * The number of tokens in each training window.
     */
    private static final int SEQUENCE_LENGTH = 512;

    /**
     * The learning rate held during the main body of the run.
     */
    private static final float PEAK_LEARNING_RATE = 3e-4f;

    /**
     * The learning rate decayed to by the final step.
     */
    private static final float FINAL_LEARNING_RATE = 3e-5f;

    /**
     * The number of steps spent ramping the learning rate up from zero. Starting at the
     * peak rate destabilises a transformer before its normalization layers have settled.
     */
    private static final int WARMUP_STEPS = 100;

    /**
     * The largest magnitude any gradient value may take.
     */
    private static final float GRADIENT_CLIP = 1f;

    /**
     * The number of steps between checkpoints.
     */
    private static final int CHECKPOINT_EVERY = 50;

    /**
     * The number of steps between progress reports.
     */
    private static final int REPORT_EVERY = 10;

    /**
     * The seed used to initialize the parameters.
     */
    private static final long PARAMETER_SEED = 42L;

    /**
     * The seed used to choose training windows.
     */
    private static final long WINDOW_SEED = 1234L;

    /**
     * The heading at which the body of the book begins, after its title and contents.
     */
    private static final String FIRST_HEADING = "## Introduction";

    /**
     * Private constructor to prevent instantiation.
     */
    private WealthOfNations() {}

    /**
     * Trains the model, checkpointing as it goes.
     * @param args The command line arguments
     * @throws Exception When the corpus or checkpoint cannot be read or written
     */
    public static void main(String[] args) throws Exception {
        int steps = intArgument(args, "--steps", 2500);
        int batchSize = intArgument(args, "--batch", BATCH_SIZE);
        boolean resume = hasFlag(args, "--resume");
        Path checkpointPath = Paths.get(stringArgument(args, "--checkpoint", "checkpoints/wealth-of-nations.ckpt"));
        Path corpusPath = Paths.get(stringArgument(args, "--corpus", ".local/corpus/the-wealth-of-nations.md"));

        String text = Corpus.load(corpusPath, FIRST_HEADING);

        LanguageModel model;
        Vocabulary vocabulary;
        int startStep = 0;

        if (resume && Files.exists(checkpointPath)) {
            Checkpoint.Restored restored = Checkpoint.load(checkpointPath).restore();
            model = restored.model();
            vocabulary = restored.vocabulary();
            startStep = restored.step();
            System.out.println("resumed from " + checkpointPath + " at step " + startStep);
        } else {
            vocabulary = new Vocabulary(text);
            model = new LanguageModel(vocabulary, FEATURES, HIDDEN_FEATURES, LAYERS, MAX_SEQUENCE, PARAMETER_SEED);
        }

        Adam optimizer = new Adam(model.parameters(), PEAK_LEARNING_RATE);
        Trainer trainer = new Trainer(model, optimizer, text, PEAK_LEARNING_RATE, GRADIENT_CLIP, WINDOW_SEED);

        System.out.println("corpus     : " + String.format("%,d", text.length()) + " characters");
        System.out.println("vocabulary : " + vocabulary.size() + " distinct tokens");
        System.out.println("parameters : " + String.format("%,d", model.parameterCount()));
        System.out.println("uniform    : " + String.format("%.4f", Math.log(vocabulary.size())) + " nats per token");
        System.out.println("training   : steps " + (startStep + 1) + " to " + steps
                + ", " + batchSize + " x " + SEQUENCE_LENGTH + " tokens each");
        System.out.println();

        long startedAt = System.currentTimeMillis();

        for (int step = startStep + 1; step <= steps; step++) {
            optimizer.learningRate(learningRateAt(step, steps));
            float loss = trainer.step(batchSize, SEQUENCE_LENGTH);

            if (step % REPORT_EVERY == 0 || step == startStep + 1) {
                double elapsed = (System.currentTimeMillis() - startedAt) / 1000.0;
                double done = step - startStep;
                double remaining = (steps - step) * elapsed / Math.max(done, 1);

                System.out.printf("step %5d/%d  loss %.4f nats  (perplexity %.2f)  lr %.2e  eta %s%n",
                        step, steps, loss, Math.exp(loss), optimizer.learningRate(), duration(remaining));
            }

            if (step % CHECKPOINT_EVERY == 0 || step == steps) {
                Checkpoint.save(checkpointPath, model, vocabulary, step);
            }
        }

        System.out.println();
        System.out.println("trained in " + duration((System.currentTimeMillis() - startedAt) / 1000.0));
        System.out.println("checkpoint " + checkpointPath.toAbsolutePath());
        System.out.println();

        Random random = new Random(7L);

        for (String prompt : new String[] {"The real price of every thing", "Labour ", "It is not from the "}) {
            System.out.println("prompt " + prompt);
            System.out.println("  " + model.generate(prompt, 200, 0.8f, 20, random).replace("\n", "\n  "));
            System.out.println();
        }
    }

    /**
     * Returns the learning rate to use at the provided step. The rate ramps linearly from
     * zero over the warmup, then follows a cosine down to its final value, which spends
     * most of the run near the peak while still arriving gently.
     * @param step The step about to be taken
     * @param steps The total number of steps in the run
     * @return The learning rate
     */
    private static float learningRateAt(int step, int steps) {
        if (step <= WARMUP_STEPS) {
            return PEAK_LEARNING_RATE * step / WARMUP_STEPS;
        }

        double progress = (double) (step - WARMUP_STEPS) / Math.max(steps - WARMUP_STEPS, 1);
        double cosine = 0.5 * (1 + Math.cos(Math.PI * Math.min(progress, 1.0)));

        return (float) (FINAL_LEARNING_RATE + (PEAK_LEARNING_RATE - FINAL_LEARNING_RATE) * cosine);
    }

    /**
     * Formats a duration in seconds as hours, minutes and seconds.
     * @param seconds The duration to format
     * @return The formatted duration
     */
    private static String duration(double seconds) {
        long total = (long) seconds;

        return String.format("%dh %02dm %02ds", total / 3600, (total % 3600) / 60, total % 60);
    }

    /**
     * Returns whether the provided flag is present in the arguments.
     * @param args The command line arguments
     * @param name The flag to look for
     * @return {@code true} if the flag is present
     */
    private static boolean hasFlag(String[] args, String name) {
        for (String arg : args) {
            if (arg.equals(name)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns the value following the provided option, or a fallback when it is absent.
     * @param args The command line arguments
     * @param name The option to look for
     * @param fallback The value to use when the option is absent
     * @return The value
     */
    private static String stringArgument(String[] args, String name, String fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }

        return fallback;
    }

    /**
     * Returns the integer value following the provided option, or a fallback when absent.
     * @param args The command line arguments
     * @param name The option to look for
     * @param fallback The value to use when the option is absent
     * @return The value
     * @throws IllegalArgumentException When the value is not an integer
     */
    private static int intArgument(String[] args, String name, int fallback) {
        String value = stringArgument(args, name, null);

        if (value == null) {
            return fallback;
        }

        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " expects an integer, got " + value + ".", e);
        }
    }
}
