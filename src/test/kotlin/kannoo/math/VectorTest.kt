package kannoo.math

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class VectorTest {
    @Test
    fun `Vector combinations require equal size`() {
        val v1 = vector(1f)
        val v2 = vector(2f, 3f)

        assertThrows<UnsupportedTensorOperation> { v1 + v2 }
        assertThrows<UnsupportedTensorOperation> { v1 - v2 }
        assertThrows<UnsupportedTensorOperation> { v1 += v2 }
        assertThrows<UnsupportedTensorOperation> { v1 -= v2 }
        assertThrows<UnsupportedTensorOperation> { v1.zip(v2) { _, _ -> 0f } }
        assertThrows<UnsupportedTensorOperation> { v1.zipSumOf(v2) { _, _ -> 0f } }
        assertThrows<UnsupportedTensorOperation> { v1.zipAssign(v2) { _, _ -> 0f } }
    }
}
