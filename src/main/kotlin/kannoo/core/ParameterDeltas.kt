package kannoo.core

import kannoo.math.Matrix
import kannoo.math.Vector

class ParameterDeltas(
    val matrices: List<ParameterDelta<Matrix>> = listOf(),
    val vectors: List<ParameterDelta<Vector>> = listOf(),
) {
    init {
        matrices.forEach {
            if (it.delta.rows != it.param.rows || it.delta.cols != it.param.cols)
                throw IllegalArgumentException("Parameter and delta matrix must have same dimensions")
        }
        vectors.forEach {
            if (it.param.size != it.delta.size)
                throw IllegalArgumentException("Parameter and delta vectors must have same size")
        }
    }
}
