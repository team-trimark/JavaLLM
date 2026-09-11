package team.trimark.javallm.llm;

import team.trimark.javallm.nn.Parameter;
import team.trimark.javallm.type.TokenMatrix;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Reads and writes the weights of a {@link LanguageModel} to disk.
 * <p>
 * A run of any length has to survive being interrupted, so training writes one of these
 * periodically and the server reads one back instead of retraining. The file carries the
 * architecture alongside the weights, because a set of weights is meaningless without the
 * shape that produced it, and it carries the vocabulary because the mapping from code
 * point to identifier must be identical or every token would be misread.
 * <p>
 * Optimizer state is deliberately not stored. Resuming therefore restarts {@link
 * team.trimark.javallm.nn.Adam}'s moment estimates from zero, which costs a brief
 * transient of a few dozen steps while they refill - far cheaper than tripling the file.
 */
public final class Checkpoint {
    /**
     * Identifies this file format, so a truncated or unrelated file is rejected loudly.
     */
    private static final int MAGIC = 0x4A4C4C4D;

    /**
     * The version of this file format.
     */
    private static final int VERSION = 1;

    /**
     * The number of features carried per position.
     */
    public final int features;

    /**
     * The width of each feed-forward hidden layer.
     */
    public final int hiddenFeatures;

    /**
     * The number of transformer blocks stacked.
     */
    public final int layers;

    /**
     * The longest sequence the model can attend over.
     */
    public final int maxSequence;

    /**
     * The seed used to initialize the parameters.
     */
    public final long seed;

    /**
     * The number of optimization steps taken before this checkpoint was written.
     */
    public final int step;

    /**
     * The text used to rebuild an identical vocabulary.
     */
    public final String vocabularyText;

    /**
     * The stored value of every parameter, keyed by name.
     */
    private final Map<String, float[]> weights;

    /**
     * Creates a checkpoint holding the provided architecture and weights.
     * @param features The number of features carried per position
     * @param hiddenFeatures The width of each feed-forward hidden layer
     * @param layers The number of transformer blocks stacked
     * @param maxSequence The longest sequence the model can attend over
     * @param seed The seed used to initialize the parameters
     * @param step The number of optimization steps taken
     * @param vocabularyText The text used to rebuild an identical vocabulary
     * @param weights The stored value of every parameter, keyed by name
     */
    private Checkpoint(int features, int hiddenFeatures, int layers, int maxSequence, long seed,
                       int step, String vocabularyText, Map<String, float[]> weights) {
        this.features = features;
        this.hiddenFeatures = hiddenFeatures;
        this.layers = layers;
        this.maxSequence = maxSequence;
        this.seed = seed;
        this.step = step;
        this.vocabularyText = vocabularyText;
        this.weights = weights;
    }

    /**
     * Writes the weights of the provided model to the provided path, atomically. The file
     * is built beside its destination and then moved into place, so a crash midway through
     * writing cannot destroy the previous checkpoint.
     * @param path The destination file
     * @param model The model to store
     * @param vocabulary The vocabulary the model was trained against
     * @param step The number of optimization steps taken so far
     * @throws NullPointerException When any argument is {@code null}
     * @throws UncheckedIOException When the file cannot be written
     */
    public static void save(Path path, LanguageModel model, Vocabulary vocabulary, int step) {
        Objects.requireNonNull(path, "Checkpoint path cannot be null.");
        Objects.requireNonNull(model, "Model cannot be null.");
        Objects.requireNonNull(vocabulary, "Vocabulary cannot be null.");

        Path temporary = path.resolveSibling(path.getFileName() + ".partial");

        try {
            Path parent = path.toAbsolutePath().getParent();

            if (parent != null) {
                Files.createDirectories(parent);
            }

            try (DataOutputStream out = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(temporary), 1 << 20))) {
                out.writeInt(MAGIC);
                out.writeInt(VERSION);
                out.writeInt(model.features());
                out.writeInt(model.hiddenFeatures());
                out.writeInt(model.layers());
                out.writeInt(model.maxSequence());
                out.writeLong(model.seed());
                out.writeInt(step);
                out.writeUTF(alphabetOf(vocabulary));

                out.writeInt(model.parameters().size());

                for (Parameter p : model.parameters()) {
                    out.writeUTF(p.name);
                    out.writeInt(p.value.rows());
                    out.writeInt(p.value.columns());

                    for (int r = 0; r < p.value.rows(); r++) {
                        for (int c = 0; c < p.value.columns(); c++) {
                            out.writeFloat(p.value.get(r, c));
                        }
                    }
                }
            }

            Files.move(temporary, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write checkpoint to " + path, e);
        }
    }

    /**
     * Reads a checkpoint from the provided path.
     * @param path The file to read
     * @return The checkpoint
     * @throws NullPointerException When the path is {@code null}
     * @throws IllegalArgumentException When the file is not a checkpoint of a known version
     * @throws UncheckedIOException When the file cannot be read
     */
    public static Checkpoint load(Path path) {
        Objects.requireNonNull(path, "Checkpoint path cannot be null.");

        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            if (in.readInt() != MAGIC) {
                throw new IllegalArgumentException(path + " is not a checkpoint file.");
            }

            int version = in.readInt();

            if (version != VERSION) {
                throw new IllegalArgumentException("Checkpoint version " + version + " is not supported.");
            }

            int features = in.readInt();
            int hiddenFeatures = in.readInt();
            int layers = in.readInt();
            int maxSequence = in.readInt();
            long seed = in.readLong();
            int step = in.readInt();
            String alphabet = in.readUTF();

            int parameterCount = in.readInt();
            Map<String, float[]> weights = new HashMap<>();

            for (int i = 0; i < parameterCount; i++) {
                String name = in.readUTF();
                int rows = in.readInt();
                int columns = in.readInt();
                float[] values = new float[rows * columns];

                for (int j = 0; j < values.length; j++) {
                    values[j] = in.readFloat();
                }

                weights.put(name, values);
            }

            return new Checkpoint(features, hiddenFeatures, layers, maxSequence, seed, step, alphabet, weights);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read checkpoint from " + path, e);
        }
    }

    /**
     * Rebuilds the model this checkpoint was taken from.
     * @return The restored model, paired with its vocabulary
     * @throws IllegalArgumentException When a stored parameter does not match the rebuilt model
     */
    public Restored restore() {
        Vocabulary vocabulary = new Vocabulary(vocabularyText);
        LanguageModel model = new LanguageModel(vocabulary, features, hiddenFeatures, layers, maxSequence, seed);

        for (Parameter p : model.parameters()) {
            float[] stored = weights.get(p.name);

            if (stored == null) {
                throw new IllegalArgumentException("Checkpoint is missing parameter " + p.name);
            }

            if (stored.length != p.count()) {
                throw new IllegalArgumentException("Parameter " + p.name + " has " + p.count()
                        + " values but the checkpoint holds " + stored.length);
            }

            TokenMatrix value = p.value;
            int index = 0;

            for (int r = 0; r < value.rows(); r++) {
                for (int c = 0; c < value.columns(); c++) {
                    value.set(r, c, stored[index++]);
                }
            }
        }

        return new Restored(model, vocabulary, step);
    }

    /**
     * Builds a string containing exactly the code points of the provided vocabulary.
     * Feeding this back to {@link Vocabulary} reproduces an identical mapping, because a
     * vocabulary is defined by the sorted set of distinct code points in its input.
     * @param vocabulary The vocabulary to describe
     * @return The alphabet
     */
    private static String alphabetOf(Vocabulary vocabulary) {
        int[] everyId = new int[vocabulary.size()];

        for (int id = 0; id < everyId.length; id++) {
            everyId[id] = id;
        }

        return vocabulary.decode(everyId);
    }

    /**
     * A model restored from a checkpoint, together with everything needed to use it.
     * @param model The restored model
     * @param vocabulary The vocabulary the model was trained against
     * @param step The number of optimization steps taken before the checkpoint was written
     */
    public record Restored(LanguageModel model, Vocabulary vocabulary, int step) {}
}
