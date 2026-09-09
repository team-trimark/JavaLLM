package team.trimark.javallm.exception;

/**
 * Thrown when no new indexes are available.
 */
public class OutOfEmptyIndexException extends RuntimeException {
    /**
     * Creates a new exception.
     * @param message The message
     */
    public OutOfEmptyIndexException(String message) {
        super(message);
    }

    /**
     * Creates a new exception.
     * @param cause The cause
     */
    public OutOfEmptyIndexException(Throwable cause) {
        super(cause);
    }

    /**
     * Creates a new exception.
     * @param message The message
     * @param cause The cause
     */
    public OutOfEmptyIndexException(String message, Throwable cause) {
        super(message, cause);
    }
}
