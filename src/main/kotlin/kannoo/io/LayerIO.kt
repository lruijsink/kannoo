package kannoo.io

import kannoo.core.InnerLayer
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.reflect.KClass

sealed class LayerIO<T : InnerLayer<*, *>>(kClass: KClass<T>) {
    val tag = kClass.tag()
    abstract fun headers(layer: T): List<String>
    abstract fun write(layer: T, outputStream: DataOutputStream)
    abstract fun read(inputStream: DataInputStream): T

    fun writeGeneric(layer: InnerLayer<*, *>, outputStream: DataOutputStream) {
        @Suppress("UNCHECKED_CAST")
        write(layer as T, outputStream)
    }
}

private val layerIO = listOf<LayerIO<*>>(
    DenseLayerIO,
)

fun DataInputStream.readLayer(): InnerLayer<*, *> {
    val type = readTerminatedString()
    return layerIO.first { it.tag == type }.read(this)
}

fun DataOutputStream.writeLayer(layer: InnerLayer<*, *>) {
    val tag = layer::class.tag()
    writeTerminatedString(tag)
    layerIO.first { it.tag == tag }.writeGeneric(layer, this)
}

private fun <T : InnerLayer<*, *>> KClass<T>.tag() = simpleName!!
