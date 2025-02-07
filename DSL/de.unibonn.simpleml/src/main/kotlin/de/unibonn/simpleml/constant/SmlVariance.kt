package de.unibonn.simpleml.constant

import de.unibonn.simpleml.simpleML.SmlTypeParameter
import de.unibonn.simpleml.simpleML.SmlTypeProjection

/**
 * The possible variances for an [SmlTypeParameter] or [SmlTypeProjection].
 */
enum class SmlVariance(val variance: String?) {

    /**
     * A complex type `G<T>` is invariant if it is neither covariant nor contravariant.
     *
     * **Example:** A `Transformer<T>` reads and writes values of type `T`. This means we cannot use a
     * `Transformer<Int>` if we want a `Transformer<Number>`, since it cannot deal with `Double` values as input.
     * Likewise, we cannot use a `Transformer<Number>` if we want a `Transformer<Int>`, since it might create `Double`
     * values as output.
     */
    Invariant(null),

    /**
     * A complex type `G<T>` is covariant if `A <= B` implies `G<A> <= G<B>`. The ordering of types is preserved.
     *
     * **Positive example:** A `Producer<T>` only ever writes values of type `T` values and never reads them. This means
     * we can use a `Producer<Int>`, where a `Producer<Number>` is expected, since `Ints` are just special `Numbers`
     * (`Int <= Number`).
     *
     * **Negative example:** A `Transformer<T>` reads and writes values of type `T`. This means we cannot use a
     * `Transformer<Int>` if we want a `Transformer<Number>`, since it cannot deal with `Double` values as input.
     */
    Covariant("out"),

    /**
     * A complex type `G<T>` is covariant if `A <= B` implies `G<B> <= G<A>`. The ordering of types is inverted.
     *
     * **Positive example:** A `Consumer<T>` only ever reads values of type `T` values and never writes them. This means
     * we can use a `Consumer<Number>`, where a `Consumer<Int>` is expected, since it is able to deal with integers
     * (`Int <= Number`).
     *
     * **Negative example:** A `Transformer<T>` reads and writes values of type `T`. This means we cannot use a
     * `Transformer<Number>` if we want a `Transformer<Int>`, since it might create `Double` values as output.
     */
    Contravariant("in");

    override fun toString(): String {
        return name
    }
}

/**
 * Returns the [SmlVariance] of this [SmlTypeParameter].
 *
 * @throws IllegalArgumentException If the variance is unknown.
 */
fun SmlTypeParameter.variance(): SmlVariance {
    return SmlVariance.values().firstOrNull { it.variance == this.variance }
        ?: throw IllegalArgumentException("Unknown variance '$variance'.")
}

/**
 * Returns the [SmlVariance] of this [SmlTypeProjection].
 *
 * @throws IllegalArgumentException If the variance is unknown.
 */
fun SmlTypeProjection.variance(): SmlVariance {
    return SmlVariance.values().firstOrNull { it.variance == this.variance }
        ?: throw IllegalArgumentException("Unknown variance '$variance'.")
}
