package kannoo.math

/**
 * Generic base class for all [Tensor] objects, to allow for operations involving the [Tensor] type to be called on a
 * generic (type-erased) instance. The following usage, for example, invokes an upcast to this generic base type:
 *
 * ```kotlin
 * fun add(a: Tensor<*>, b: Tensor<*>) = a + b
 * ```
 *
 * Generic methods are defined on [Tensor] itself.
 */
sealed interface TensorBase
