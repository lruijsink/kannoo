package kannoo.math

import kotlin.math.exp
import kotlin.math.log2

/**
 * Variant of [Tensor.transform] which preserves the receiver type [T].
 *
 * @param function Function to apply
 *
 * @return A copy of the tensor with [function] applied to each scalar element
 */
inline fun <T : Tensor> T.transformGeneric(crossinline function: (Float) -> Float): T {
    @Suppress("UNCHECKED_CAST")
    return this.transform { function(it) } as T
}

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor with `exp` applied to all scalar elements
 */
fun <T : Tensor> exp(tensor: T): T = tensor.transformGeneric(::exp)

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor with `log2` applied to all scalar elements
 */
fun <T : Tensor> log2(tensor: T): T = tensor.transformGeneric(::log2) as T

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor all scalar elements squared
 */
fun <T : Tensor> square(tensor: T): T = tensor.transformGeneric { it * it } as T
