package de.unibonn.simpleml.serializer

import com.google.inject.Inject
import org.eclipse.emf.ecore.EObject
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.resource.SaveOptions
import org.eclipse.xtext.serializer.impl.Serializer

internal object SerializerExtensionsInjectionTarget {

    @Inject
    lateinit var serializer: Serializer
}

/**
 * Serializes a subtree of the EMF model and applies the formatter to it. This only works if the [EObject] is part of a
 * [Resource].
 *
 * @receiver The root of the subtree.
 * @return A result object indicating success or failure.
 */
fun EObject.serializeToFormattedString(): SerializationResult {
    return serializeToStringWithSaveOptions(WithFormatting)
}

private val WithoutFormatting = SaveOptions.defaultOptions()
private val WithFormatting = SaveOptions.newBuilder().format().options

private fun EObject.serializeToStringWithSaveOptions(options: SaveOptions): SerializationResult {
    if (this.eResource() == null) {
        return SerializationResult.NotInResourceFailure
    }

    return try {
        val code = SerializerExtensionsInjectionTarget.serializer
            .serialize(this, options)
            .trim()
            .replace(System.lineSeparator(), "\n")

        SerializationResult.Success(code)
    } catch (e: RuntimeException) {
        SerializationResult.WrongEmfModelStructureFailure(e.message ?: "")
    }
}

/**
 * Result of calling [serializeToString] or [serializeToFormattedString].
 */
sealed interface SerializationResult {

    /**
     * Serialization was successful.
     *
     * @param code The created DSL code.
     */
    class Success(val code: String) : SerializationResult

    /**
     * Something went wrong while serializing the [EObject].
     */
    sealed interface Failure : SerializationResult {

        /**
         * A message that describes the failure.
         */
        val message: String
    }

    /**
     * The [EObject] is not part of a [Resource] and cannot be serialized.
     */
    object NotInResourceFailure : Failure {
        override val message: String
            get() = "The EObject is not part of a Resource and cannot be serialized."
    }

    /**
     * The EMF model is not configured correctly.
     */
    class WrongEmfModelStructureFailure(override val message: String) : Failure
}