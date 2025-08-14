package kannoo.math

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class MatrixTest {

    @Test
    fun `Matrices cannot be created with unequal column widths`() {
        assertThrows<IncompatibleShapeException> { matrix(vector(1f, 2f, 3f), vector(4f, 5f)) }
        assertThrows<IncompatibleShapeException> { matrix(vector(1f, 2f, 3f), vector()) }
        assertThrows<IncompatibleShapeException> { Matrix(arrayOf(vector(1f, 2f), vector(3f))) }
    }
}
