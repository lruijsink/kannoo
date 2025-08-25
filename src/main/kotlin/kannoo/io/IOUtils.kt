package kannoo.io

import kannoo.core.Model
import kannoo.math.Dimensions
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Tensor3
import kannoo.math.Tensor4
import kannoo.math.Vector
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.FileInputStream
import java.io.FileOutputStream

private const val STRING_TERMINATOR = '\n'

fun DataInputStream.readVector(size: Int): Vector =
    Vector(size) { readFloat() }

fun DataInputStream.readMatrix(rows: Int, cols: Int): Matrix =
    Matrix(Array(rows) { readVector(cols) })

fun DataInputStream.readTensor3(size: Int, sliceRows: Int, sliceCols: Int): Tensor3 =
    Tensor3(size) { readMatrix(sliceRows, sliceCols) }

fun DataInputStream.readTensor4(size: Int, subSlices: Int, sliceRows: Int, sliceCols: Int): Tensor4 =
    Tensor4(size) { readTensor3(subSlices, sliceRows, sliceCols) }

fun DataOutputStream.writeVector(vector: Vector) {
    vector.elements.forEach { writeFloat(it) }
}

fun DataOutputStream.writeMatrix(matrix: Matrix) {
    matrix.rowVectors.forEach { writeVector(it) }
}

fun DataOutputStream.writeTensor3(tensor: Tensor3) {
    tensor.slices.forEach { writeMatrix(it) }
}

fun DataOutputStream.writeTensor4(tensor: Tensor4) {
    tensor.slices.forEach { writeTensor3(it) }
}

fun DataInputStream.readTerminatedString(): String {
    var res = ""
    while (true) {
        val c = readChar()
        if (c == STRING_TERMINATOR) break
        res += c
    }
    return res
}

fun DataOutputStream.writeTerminatedString(value: String) {
    writeChars(value)
    writeChar(STRING_TERMINATOR.code)
}

fun writeModelToFile(model: Model, fileName: String) {
    BufferedOutputStream(FileOutputStream(fileName)).use { it.writeModel(model) }
}

fun readModelFromFile(fileName: String): Model =
    BufferedInputStream(FileInputStream(fileName)).use { it.readModel() }

fun DataOutputStream.writeShape(shape: Shape) {
    writeInt(shape.rank)
    shape.dimensions.forEach { writeInt(it) }
}

fun DataInputStream.readShape(): Shape {
    val rank = readInt()
    return Shape(List(rank) { readInt() })
}

fun DataOutputStream.writeDimensions(dimensions: Dimensions) {
    writeInt(dimensions.height)
    writeInt(dimensions.width)
}

fun DataInputStream.readDimensions(): Dimensions =
    Dimensions(readInt(), readInt())

fun <T : Any> DataOutputStream.writeNullable(value: T?, doWrite: DataOutputStream.(T) -> Unit) {
    writeBoolean(value != null)
    if (value != null) doWrite(value)
}

fun <T : Any> DataInputStream.readNullable(doRead: DataInputStream.() -> T): T? =
    if (readBoolean()) doRead()
    else null
