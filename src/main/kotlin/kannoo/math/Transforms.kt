package kannoo.math

import kotlin.math.exp
import kotlin.math.log2

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor with `exp` applied to all scalar elements
 */
fun <T : Tensor<T>> exp(tensor: T): T =
    tensor.map(::exp)

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor with `log2` applied to all scalar elements
 */
fun <T : Tensor<T>> log2(tensor: T): T =
    tensor.map(::log2)

/**
 * @param tensor The tensor to transform
 *
 * @return A new tensor all scalar elements squared
 */
fun <T : Tensor<T>> square(tensor: T): T =
    tensor.map { it * it }
