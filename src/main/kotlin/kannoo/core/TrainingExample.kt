package kannoo.core

import kannoo.math.Vector

data class TrainingExample(
    val input: Vector,
    val target: Vector,
)