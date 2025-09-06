package kannoo.vulkan

import org.lwjgl.system.MemoryStack
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun interface VulkanPushConstants {
    fun record(stack: MemoryStack): ByteBuffer
}

fun pushConstants(vararg values: Any) = VulkanPushConstants { stack ->
    val size = 4 * values.size // TODO: Make this more type-flexible
    val data = stack.malloc(size).order(ByteOrder.nativeOrder())
    for (value in values) {
        when (value) {
            is Int -> data.putInt(value)
            is Float -> data.putFloat(value)
            is Boolean -> data.putInt(if (value) 1 else 0)
            else -> throw IllegalStateException("Push constant $value (${value::class}) not supported yet")
        }
    }
    data.flip()
    data
}
