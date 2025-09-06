package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.vkAllocateDescriptorSets
import org.lwjgl.vulkan.VK10.vkCreateDescriptorPool
import org.lwjgl.vulkan.VK10.vkCreateDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorPool
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkUpdateDescriptorSets
import org.lwjgl.vulkan.VkDescriptorBufferInfo
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo
import org.lwjgl.vulkan.VkDescriptorPoolSize
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo
import org.lwjgl.vulkan.VkWriteDescriptorSet

class VulkanDescriptorSet(
    vulkan: Vulkan,
    val bufferBindings: Map<Int, VulkanBuffer>,
) : VulkanResource(vulkan) {

    val layout: Long = createDescriptorSetLayout()
    val pool: Long = createDescriptorPool()
    val handle: Long = createDescriptorSet()

    private fun createDescriptorSetLayout(): Long = stackPush().use { stack ->
        val bindings = VkDescriptorSetLayoutBinding.calloc(bufferBindings.size)

        bufferBindings.keys.forEachIndexed { i, binding ->
            bindings.get(i)
                .binding(binding)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .descriptorCount(1)
                .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
        }

        val createInfo = VkDescriptorSetLayoutCreateInfo.calloc()
            .`sType$Default`()
            .pBindings(bindings)

        val pDescriptorSetLayout = stack.mallocLong(1)
        vkCreateDescriptorSetLayout(vulkan.device, createInfo, null, pDescriptorSetLayout).orThrow()
        return pDescriptorSetLayout.get(0)
    }

    private fun createDescriptorPool(): Long = stackPush().use { stack ->
        val poolSize = VkDescriptorPoolSize.calloc(1)
            .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(bufferBindings.size)

        val poolCreateInfo = VkDescriptorPoolCreateInfo.calloc()
            .`sType$Default`()
            .maxSets(1)
            .pPoolSizes(poolSize)

        val pDescriptorPool = stack.mallocLong(1)
        vkCreateDescriptorPool(vulkan.device, poolCreateInfo, null, pDescriptorPool).orThrow()
        return pDescriptorPool.get()
    }

    private fun createDescriptorSet(): Long = stackPush().use { stack ->
        val allocateInfo = VkDescriptorSetAllocateInfo.calloc()
            .`sType$Default`()
            .descriptorPool(pool)
            .pSetLayouts(stack.longs(layout))

        val pDescriptorSets = stack.mallocLong(1)
        vkAllocateDescriptorSets(vulkan.device, allocateInfo, pDescriptorSets).orThrow()
        val descriptorSet = pDescriptorSets.get()

        val writeDescriptorSet = VkWriteDescriptorSet.calloc(bufferBindings.size)

        bufferBindings.entries.forEachIndexed { i, (binding, buffer) ->
            val bufferInfo = VkDescriptorBufferInfo.calloc(1)
                .buffer(buffer.handle)
                .offset(0)
                .range(buffer.size)

            writeDescriptorSet.get(i)
                .`sType$Default`()
                .dstSet(descriptorSet)
                .dstBinding(binding)
                .descriptorCount(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .pBufferInfo(bufferInfo)
        }

        vkUpdateDescriptorSets(vulkan.device, writeDescriptorSet, null)

        return descriptorSet
    }

    override fun destroy() {
        vkDestroyDescriptorPool(vulkan.device, pool, null)
        vkDestroyDescriptorSetLayout(vulkan.device, layout, null)
    }
}

fun Vulkan.createDescriptorSet(vararg bufferBindings: Pair<Int, VulkanBuffer>): VulkanDescriptorSet =
    VulkanDescriptorSet(this, mapOf(*bufferBindings))
