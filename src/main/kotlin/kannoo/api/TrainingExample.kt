package kannoo.api

import kannoo.Vector

data class TrainingExample(
    val input: Vector,
    val target: Vector,
)
