package kannoo.core

import kannoo.math.Vector

class BackPropagation(
    /**
     * Delta (error term) of the (activated) input = derivative of the cost w.r.t. this layer's inputs
     */
    val deltaInput: Vector,

    val parameterDeltas: ParameterDeltas,
)
