package kannoo.core

fun interface InnerLayerInitializer<T : InnerLayer> {
    fun initialize(previousLayerSize: Int): T
}
