package kannoo.vulkan

import kannoo.math.randomSignedFloat
import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.system.MemoryUtil.memAddress
import org.lwjgl.system.MemoryUtil.memByteBuffer
import org.lwjgl.system.MemoryUtil.memSet
import org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
import org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE
import org.lwjgl.vulkan.VK10.vkAllocateMemory
import org.lwjgl.vulkan.VK10.vkBindBufferMemory
import org.lwjgl.vulkan.VK10.vkCreateBuffer
import org.lwjgl.vulkan.VK10.vkDestroyBuffer
import org.lwjgl.vulkan.VK10.vkFreeMemory
import org.lwjgl.vulkan.VK10.vkGetBufferMemoryRequirements
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VK10.vkMapMemory
import org.lwjgl.vulkan.VK10.vkUnmapMemory
import org.lwjgl.vulkan.VkBufferCreateInfo
import org.lwjgl.vulkan.VkMemoryAllocateInfo
import org.lwjgl.vulkan.VkMemoryRequirements
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties
import java.nio.ByteBuffer

class VulkanBuffer(
    vulkan: Vulkan,
    val size: Long,
) : VulkanResource(vulkan) {

    val handle: Long = createBuffer()
    val memory: Long = createMemory()
    val mapped: ByteBuffer = map()

    private fun createBuffer(): Long = stackPush().use { stack ->
        val createInfo = VkBufferCreateInfo.calloc()
            .`sType$Default`()
            .size(size)
            .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            .sharingMode(VK_SHARING_MODE_EXCLUSIVE)

        val pBuffer = stack.mallocLong(1)
        vkCreateBuffer(vulkan.device, createInfo, null, pBuffer).orThrow()
        return pBuffer.get()
    }

    private fun createMemory(): Long = stackPush().use { stack ->
        val memoryRequirements = VkMemoryRequirements.calloc()
        vkGetBufferMemoryRequirements(vulkan.device, handle, memoryRequirements)

        val allocateInfo = VkMemoryAllocateInfo.calloc()
            .`sType$Default`()
            .allocationSize(memoryRequirements.size())
            .memoryTypeIndex(
                findMemoryType(
                    memoryRequirements.memoryTypeBits(),
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT or VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                )
            )

        val pMemory = stack.mallocLong(1)
        vkAllocateMemory(vulkan.device, allocateInfo, null, pMemory).orThrow()
        val memory = pMemory.get()

        vkBindBufferMemory(vulkan.device, handle, memory, 0).orThrow()

        return memory
    }

    private fun findMemoryType(memoryTypeBits: Int, properties: Int): Int = stackPush().use { stack ->
        val memoryProperties = VkPhysicalDeviceMemoryProperties.calloc()
        vkGetPhysicalDeviceMemoryProperties(vulkan.physicalDevice, memoryProperties)

        for (i in 0 until memoryProperties.memoryTypeCount())
            if (memoryTypeBits and (1 shl i) != 0 &&
                (memoryProperties.memoryTypes().get(i).propertyFlags() and properties) == properties
            ) return i

        return -1
    }

    private fun map(): ByteBuffer = stackPush().use { stack ->
        val pMappedMemory = stack.mallocPointer(1)
        vkMapMemory(vulkan.device, memory, 0, size, 0, pMappedMemory).orThrow()
        memByteBuffer(pMappedMemory.get(0), size.toInt())
    }

    fun initZero() {
        memSet(memAddress(mapped), 0, mapped.capacity() * Float.SIZE_BYTES.toLong())
    }

    fun initRandom() {
        val floats = mapped.asFloatBuffer()
        repeat(floats.capacity()) {
            floats.put(randomSignedFloat())
        }
        floats.flip()
    }

    override fun destroy() {
        vkUnmapMemory(vulkan.device, memory)
        vkFreeMemory(vulkan.device, memory, null)
        vkDestroyBuffer(vulkan.device, handle, null)
    }
}

fun Vulkan.createBuffer(size: Long): VulkanBuffer =
    VulkanBuffer(this, size)
