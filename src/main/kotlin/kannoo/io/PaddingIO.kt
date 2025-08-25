package kannoo.io

import kannoo.math.CircularPadding
import kannoo.math.Padding
import kannoo.math.ReflectionPadding
import kannoo.math.ReplicationPadding
import kannoo.math.ZeroPadding
import java.io.DataInputStream
import java.io.DataOutputStream

private val paddingSchemes = listOf(
    ZeroPadding,
    CircularPadding,
    ReflectionPadding,
    ReplicationPadding,
)

fun DataInputStream.readPadding(): Padding {
    val schemeTag = readTerminatedString()
    val scheme = paddingSchemes.find { it::class.simpleName == schemeTag }
        ?: throw IllegalStateException("Unknown padding scheme '$schemeTag'")

    return Padding(readInt(), readInt(), scheme)
}

fun DataOutputStream.writePadding(padding: Padding) {
    writeTerminatedString(padding.scheme::class.simpleName!!)
    writeInt(padding.height)
    writeInt(padding.width)
}
