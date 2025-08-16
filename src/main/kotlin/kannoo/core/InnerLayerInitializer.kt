package kannoo.core

import kannoo.math.Shape

fun interface InnerLayerInitializer<T : InnerLayer<*, *>> {
    fun initialize(previousLayerShape: Shape): T
}
