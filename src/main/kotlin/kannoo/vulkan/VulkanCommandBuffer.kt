package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY
import org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers
import org.lwjgl.vulkan.VK10.vkBeginCommandBuffer
import org.lwjgl.vulkan.VK10.vkCmdBindDescriptorSets
import org.lwjgl.vulkan.VK10.vkCmdBindPipeline
import org.lwjgl.vulkan.VK10.vkCmdDispatch
import org.lwjgl.vulkan.VK10.vkCmdPushConstants
import org.lwjgl.vulkan.VK10.vkCreateCommandPool
import org.lwjgl.vulkan.VK10.vkDestroyCommandPool
import org.lwjgl.vulkan.VK10.vkEndCommandBuffer
import org.lwjgl.vulkan.VkCommandBuffer
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo
import org.lwjgl.vulkan.VkCommandBufferBeginInfo
import org.lwjgl.vulkan.VkCommandPoolCreateInfo
import java.nio.IntBuffer

class VulkanCommandBuffer(
    val vulkan: Vulkan,
    val pipeline: VulkanPipeline,
    val groupCountX: Int,
    val groupCountY: Int,
    val groupCountZ: Int,
) {
    val pool: Long = createCommandPool()
    val handle: VkCommandBuffer = createCommandBuffer()

    private fun createCommandPool(): Long = stackPush().use { stack ->
        val commandPoolCreateInfo = VkCommandPoolCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)
            .queueFamilyIndex(vulkan.queueFamilyIndex)

        val pCommandPool = stack.mallocLong(1)
        vkCreateCommandPool(vulkan.device, commandPoolCreateInfo, null, pCommandPool).orThrow()
        return pCommandPool.get()
    }

    private fun createCommandBuffer(): VkCommandBuffer = stackPush().use { stack ->
        val commandBufferAllocateInfo = VkCommandBufferAllocateInfo.calloc()
            .`sType$Default`()
            .commandPool(pool)
            .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
            .commandBufferCount(1)

        val pCommandBuffer = stack.mallocPointer(1)
        vkAllocateCommandBuffers(vulkan.device, commandBufferAllocateInfo, pCommandBuffer).orThrow()
        val commandBuffer = VkCommandBuffer(pCommandBuffer.get(), vulkan.device)

        val beginInfo = VkCommandBufferBeginInfo.calloc()
            .`sType$Default`()

        vkBeginCommandBuffer(commandBuffer, beginInfo).orThrow()

        vkCmdPushConstants(
            commandBuffer,
            pipeline.layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            stack.ints(*pipeline.pushConstants.values.toIntArray()),
        )

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout,
            0,
            stack.longs(pipeline.descriptorSet.handle),
            null as IntBuffer?,
        )
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ)
        vkEndCommandBuffer(commandBuffer).orThrow()

        return commandBuffer
    }

    fun destroy() {
        vkDestroyCommandPool(vulkan.device, pool, null)
    }
}

fun Vulkan.createCommandBuffer(
    pipeline: VulkanPipeline,
    groupCountX: Int,
    groupCountY: Int = 1,
    groupCountZ: Int = 1,
): VulkanCommandBuffer =
    VulkanCommandBuffer(this, pipeline, groupCountX, groupCountY, groupCountZ)
