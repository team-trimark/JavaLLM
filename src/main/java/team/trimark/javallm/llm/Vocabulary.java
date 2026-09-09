package team.trimark.javallm.llm;

import team.trimark.javallm.tokenization.Detokenizer;
import team.trimark.javallm.tokenization.Tokenizer;
import team.trimark.javallm.type.TokenArray;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;

/**
 * A bidirectional mapping between the tokens produced by a {@link Tokenizer} and the contiguous
 * integer identifiers a model needs. A model's output layer has one entry per vocabulary item, so
 * the identifiers must be dense, whereas raw code points are sparse.
 */
public final class Vocabulary {
    /**
     * The tokenizer used to convert text into tokens.
     */
    private final Tokenizer tokenizer = new CodePointTokenizer();

    /**
     * The detokenizer used to convert tokens back into text.
     */
    private final Detokenizer detokenizer = new CodePointDetokenizer();

    /**
     * The mapping from code point to dense identifier.
     */
    private final Map<Integer, Integer> codePointToId = new HashMap<>();

    /**
     * The mapping from dense identifier to code point.
     */
    private final int[] idToCodePoint;

    /**
     * Builds a vocabulary containing every distinct code point of the provided text.
     * @param text The text to learn the vocabulary from
     * @throws NullPointerException When the text is {@code null}
     */
    public Vocabulary(String text) {
        Objects.requireNonNull(text, "Vocabulary text cannot be null.");

        TreeSet<Integer> distinct = new TreeSet<>();

        for (float token : tokenizer.apply(text)) {
            distinct.add(Float.floatToRawIntBits(token));
        }

        this.idToCodePoint = new int[distinct.size()];
        int next = 0;

        for (int codePoint : distinct) {
            idToCodePoint[next] = codePoint;
            codePointToId.put(codePoint, next);
            next++;
        }
    }

    /**
     * Returns the number of distinct tokens in this vocabulary.
     * @return The size of this vocabulary
     */
    public int size() {
        return idToCodePoint.length;
    }

    /**
     * Converts text into a sequence of dense identifiers, skipping any code point which is not part
     * of this vocabulary.
     * @param text The text to encode
     * @return The identifiers
     */
    public int[] encode(String text) {
        TokenArray tokens = tokenizer.apply(text);
        int[] ids = new int[tokens.size()];
        int count = 0;

        for (int i = 0; i < tokens.size(); i++) {
            Integer id = codePointToId.get(Float.floatToRawIntBits(tokens.get(i)));

            if (id != null) {
                ids[count++] = id;
            }
        }

        int[] trimmed = new int[count];
        System.arraycopy(ids, 0, trimmed, 0, count);
        return trimmed;
    }

    /**
     * Converts a sequence of dense identifiers back into text.
     * @param ids The identifiers to decode
     * @return The decoded text
     * @throws IndexOutOfBoundsException When an identifier is outside this vocabulary
     */
    public String decode(int[] ids) {
        TokenArray tokens = new TokenArray(ids.length);

        for (int id : ids) {
            tokens.append(Float.intBitsToFloat(idToCodePoint[id]));
        }

        return detokenizer.apply(tokens);
    }

    /**
     * Returns a printable description of the token with the provided identifier.
     * @param id The identifier to describe
     * @return The description
     * @throws IndexOutOfBoundsException When the identifier is outside this vocabulary
     */
    public String describe(int id) {
        int codePoint = idToCodePoint[id];
        return codePoint == '\n' ? "\\n" : new String(Character.toChars(codePoint));
    }
}
