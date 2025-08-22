package kannoo.core

import kannoo.math.Shape

class InputLayer(val shape: Shape) {
    constructor(size: Int) : this(Shape(size))
}
