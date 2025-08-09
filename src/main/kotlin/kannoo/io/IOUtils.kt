package kannoo.io

import kannoo.core.Model
import kannoo.math.Matrix
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

fun DataOutputStream.writeVector(vector: Vector) {
    vector.elements.forEach { writeFloat(it) }
}

fun DataOutputStream.writeMatrix(matrix: Matrix) {
    matrix.rowVectors.forEach { writeVector(it) }
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
