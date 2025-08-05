package kannoo.core

import kannoo.math.Matrix
import kannoo.math.Vector

class ParameterDeltas(
    val matrices: List<ParameterDelta<Matrix>> = listOf(),
    val vectors: List<ParameterDelta<Vector>> = listOf(),
)
