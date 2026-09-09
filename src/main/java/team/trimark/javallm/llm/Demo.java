package team.trimark.javallm.llm;

import team.trimark.javallm.nn.Adam;

import java.util.Random;

/**
 * Trains a small language model on a short corpus and then samples from it, so the whole pipeline
 * can be watched end to end.
 */
public final class Demo {
    /**
     * The text the model learns from. It is deliberately small and highly patterned, because a model
     * of this size can only discover structure that repeats often.
     */
    private static final String CORPUS = """
            the cat sat on the mat.
            the cat sat on the hat.
            the dog sat on the mat.
            the dog sat on the log.
            a cat ran to the mat.
            a dog ran to the log.
            the cat ran to the hat.
            the dog ran to the mat.
            a cat sat on the log.
            a dog sat on the hat.
            """.repeat(12);

    /**
     * The number of features carried per position.
     */
    private static final int FEATURES = 48;

    /**
     * The width of each feed-forward hidden layer.
     */
    private static final int HIDDEN_FEATURES = 128;

    /**
     * The number of transformer blocks stacked.
     */
    private static final int LAYERS = 2;

    /**
     * The longest sequence the model can attend over.
     */
    private static final int MAX_SEQUENCE = 32;

    /**
     * The number of optimization steps to run.
     */
    private static final int STEPS = 400;

    /**
     * The number of windows averaged into each step.
     */
    private static final int BATCH_SIZE = 8;

    /**
     * The number of tokens in each training window.
     */
    private static final int SEQUENCE_LENGTH = 32;

    /**
     * The learning rate handed to the optimizer.
     */
    private static final float LEARNING_RATE = 3e-3f;

    /**
     * The largest magnitude any gradient value may take.
     */
    private static final float GRADIENT_CLIP = 1f;

    /**
     * Private constructor to prevent instantiation.
     */
    private Demo() {}

    /**
     * Trains the model and prints samples drawn from it.
     * @param args The command line arguments, which are ignored
     */
    public static void main(String[] args) {
        Vocabulary vocabulary = new Vocabulary(CORPUS);
        LanguageModel model = new LanguageModel(vocabulary, FEATURES, HIDDEN_FEATURES, LAYERS, MAX_SEQUENCE, 42L);
        Adam optimizer = new Adam(model.parameters(), LEARNING_RATE);
        Trainer trainer = new Trainer(model, optimizer, CORPUS, LEARNING_RATE, GRADIENT_CLIP, 1234L);

        System.out.println("vocabulary  : " + vocabulary.size() + " distinct tokens");
        System.out.println("parameters  : " + model.parameterCount());
        System.out.println("uniform loss: " + String.format("%.4f", Math.log(vocabulary.size())) + " nats per token");
        System.out.println();

        long startedAt = System.currentTimeMillis();

        for (int step = 1; step <= STEPS; step++) {
            float loss = trainer.step(BATCH_SIZE, SEQUENCE_LENGTH);

            if (step % 25 == 0 || step == 1) {
                System.out.printf("step %4d/%d  loss %.4f nats  (perplexity %.2f)%n",
                        step, STEPS, loss, Math.exp(loss));
            }
        }

        System.out.printf("%ntrained in %.1f s%n%n", (System.currentTimeMillis() - startedAt) / 1000.0);

        Random random = new Random(7L);

        for (String prompt : new String[] {"the cat ", "a dog ", "the "}) {
            String continuation = model.generate(prompt, 28, 0.7f, 5, random);
            System.out.println("prompt " + quote(prompt) + " -> " + quote(continuation.replace("\n", "\\n")));
        }
    }

    /**
     * Wraps a string in quotation marks so trailing spaces stay visible.
     * @param text The text to wrap
     * @return The quoted text
     */
    private static String quote(String text) {
        return "\"" + text + "\"";
    }
}
