package team.trimark.javallm.llm;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Loads a training corpus from a markdown file on disk.
 * <p>
 * The front matter of a transcribed book - its title block, provenance note and table of
 * contents - is navigational rather than prose. It is a few thousand characters of link
 * syntax that teaches a character level model nothing about the language of the body, so
 * this loader drops everything before the first body heading.
 */
public final class Corpus {
    /**
     * Private constructor to prevent instantiation.
     */
    private Corpus() {}

    /**
     * Reads a corpus from the provided path, dropping any front matter that precedes the
     * first occurrence of the provided heading.
     * @param path The file to read
     * @param firstHeading The heading which begins the body text
     * @return The body text
     * @throws NullPointerException When either argument is {@code null}
     * @throws IllegalArgumentException When the file is empty, or the heading is not found
     * @throws UncheckedIOException When the file cannot be read
     */
    public static String load(Path path, String firstHeading) {
        Objects.requireNonNull(path, "Corpus path cannot be null.");
        Objects.requireNonNull(firstHeading, "Heading cannot be null.");

        String text;

        try {
            text = Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read corpus at " + path, e);
        }

        if (text.isEmpty()) {
            throw new IllegalArgumentException("Corpus at " + path + " is empty.");
        }

        int start = text.indexOf(firstHeading);

        if (start < 0) {
            throw new IllegalArgumentException("Heading \"" + firstHeading + "\" not found in " + path);
        }

        return normalise(text.substring(start));
    }

    /**
     * Collapses carriage returns and runs of blank lines, so the model does not spend
     * capacity on line ending trivia that carries no meaning.
     * @param text The text to normalise
     * @return The normalised text
     */
    private static String normalise(String text) {
        String unified = text.replace("\r\n", "\n").replace('\r', '\n');

        while (unified.contains("\n\n\n")) {
            unified = unified.replace("\n\n\n", "\n\n");
        }

        return unified;
    }
}
