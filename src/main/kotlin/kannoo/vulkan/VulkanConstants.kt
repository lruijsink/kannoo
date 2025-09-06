package kannoo.vulkan

import org.lwjgl.system.MemoryStack
import org.lwjgl.vulkan.VkSpecializationInfo
import org.lwjgl.vulkan.VkSpecializationMapEntry
import java.nio.ByteOrder

fun interface VulkanSpecialization {
    fun createInfo(stack: MemoryStack): VkSpecializationInfo
}

fun createSpecialization(entries: Map<Int, Any>) = VulkanSpecialization { stack ->
    val sizes = entries.mapValues { (_, value) ->
        when (value) {
            is Int -> Int.SIZE_BYTES
            is Float -> Float.SIZE_BYTES
            is Boolean -> Int.SIZE_BYTES
            else -> throw IllegalStateException("Cannot use $value (${value::class}) as constant")
        }
    }
    val mapEntries = VkSpecializationMapEntry.calloc(entries.size, stack)
    val data = stack.malloc(sizes.values.sum()).order(ByteOrder.nativeOrder())

    var offset = 0
    entries.entries.forEachIndexed { i, (id, value) ->
        val size = sizes[id]!!
        mapEntries.get(i).constantID(id).offset(offset).size(size.toLong())
        offset += size

        when (value) {
            is Int -> data.putInt(value)
            is Float -> data.putFloat(value)
            is Boolean -> data.putInt(if (value) 1 else 0)
        }
    }
    data.flip()
    VkSpecializationInfo.calloc(stack).pMapEntries(mapEntries).pData(data)
}

fun createSpecialization(vararg entries: Pair<Int, Any>): VulkanSpecialization =
    createSpecialization(entries.toMap())
