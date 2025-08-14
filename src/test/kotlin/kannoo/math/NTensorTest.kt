package kannoo.math

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class NTensorTest {

    @Test
    fun `NTensor shape is highest rank to lowest`() {
        assertEquals(
            listOf(2, 3, 4),
            tensor(
                matrix(
                    vector(1f, 2f, 3f, 4f),
                    vector(5f, 6f, 7f, 8f),
                    vector(9f, 0f, 1f, 2f),
                ),
                matrix(
                    vector(3f, 4f, 5f, 6f),
                    vector(7f, 8f, 9f, 0f),
                    vector(1f, 2f, 3f, 4f),
                ),
            ).shape
        )
    }
}
