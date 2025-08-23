package kannoo.core

import kannoo.math.Shape

fun interface InnerLayerInitializer {
    fun initialize(previousLayerShape: Shape): InnerLayer
}
