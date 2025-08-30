package kannoo.vulkan

import org.lwjgl.PointerBuffer
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.system.MemoryUtil.memByteBuffer
import org.lwjgl.system.MemoryUtil.memFree
import org.lwjgl.vulkan.EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME
import org.lwjgl.vulkan.KHRGetPhysicalDeviceProperties2.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
import org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
import org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
import org.lwjgl.vulkan.KHRPortabilitySubset.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
import org.lwjgl.vulkan.VK10.VK_API_VERSION_1_0
import org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
import org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY
import org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
import org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
import org.lwjgl.vulkan.VK10.VK_NULL_HANDLE
import org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE
import org.lwjgl.vulkan.VK10.VK_QUEUE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE
import org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_APPLICATION_INFO
import org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
import org.lwjgl.vulkan.VK10.VK_VERSION_MAJOR
import org.lwjgl.vulkan.VK10.VK_VERSION_MINOR
import org.lwjgl.vulkan.VK10.VK_VERSION_PATCH
import org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers
import org.lwjgl.vulkan.VK10.vkAllocateDescriptorSets
import org.lwjgl.vulkan.VK10.vkAllocateMemory
import org.lwjgl.vulkan.VK10.vkBeginCommandBuffer
import org.lwjgl.vulkan.VK10.vkBindBufferMemory
import org.lwjgl.vulkan.VK10.vkCmdBindDescriptorSets
import org.lwjgl.vulkan.VK10.vkCmdBindPipeline
import org.lwjgl.vulkan.VK10.vkCmdDispatch
import org.lwjgl.vulkan.VK10.vkCreateBuffer
import org.lwjgl.vulkan.VK10.vkCreateCommandPool
import org.lwjgl.vulkan.VK10.vkCreateComputePipelines
import org.lwjgl.vulkan.VK10.vkCreateDescriptorPool
import org.lwjgl.vulkan.VK10.vkCreateDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkCreateDevice
import org.lwjgl.vulkan.VK10.vkCreateFence
import org.lwjgl.vulkan.VK10.vkCreateInstance
import org.lwjgl.vulkan.VK10.vkCreatePipelineLayout
import org.lwjgl.vulkan.VK10.vkCreateShaderModule
import org.lwjgl.vulkan.VK10.vkDestroyBuffer
import org.lwjgl.vulkan.VK10.vkDestroyCommandPool
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorPool
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkDestroyDevice
import org.lwjgl.vulkan.VK10.vkDestroyFence
import org.lwjgl.vulkan.VK10.vkDestroyInstance
import org.lwjgl.vulkan.VK10.vkDestroyPipeline
import org.lwjgl.vulkan.VK10.vkDestroyPipelineLayout
import org.lwjgl.vulkan.VK10.vkDestroyShaderModule
import org.lwjgl.vulkan.VK10.vkEndCommandBuffer
import org.lwjgl.vulkan.VK10.vkEnumerateInstanceExtensionProperties
import org.lwjgl.vulkan.VK10.vkEnumerateInstanceLayerProperties
import org.lwjgl.vulkan.VK10.vkEnumeratePhysicalDevices
import org.lwjgl.vulkan.VK10.vkFreeMemory
import org.lwjgl.vulkan.VK10.vkGetBufferMemoryRequirements
import org.lwjgl.vulkan.VK10.vkGetDeviceQueue
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceProperties
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceQueueFamilyProperties
import org.lwjgl.vulkan.VK10.vkMapMemory
import org.lwjgl.vulkan.VK10.vkQueueSubmit
import org.lwjgl.vulkan.VK10.vkUnmapMemory
import org.lwjgl.vulkan.VK10.vkUpdateDescriptorSets
import org.lwjgl.vulkan.VK10.vkWaitForFences
import org.lwjgl.vulkan.VkApplicationInfo
import org.lwjgl.vulkan.VkBufferCreateInfo
import org.lwjgl.vulkan.VkCommandBuffer
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo
import org.lwjgl.vulkan.VkCommandBufferBeginInfo
import org.lwjgl.vulkan.VkCommandPoolCreateInfo
import org.lwjgl.vulkan.VkComputePipelineCreateInfo
import org.lwjgl.vulkan.VkDescriptorBufferInfo
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo
import org.lwjgl.vulkan.VkDescriptorPoolSize
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo
import org.lwjgl.vulkan.VkDevice
import org.lwjgl.vulkan.VkDeviceCreateInfo
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo
import org.lwjgl.vulkan.VkExtensionProperties
import org.lwjgl.vulkan.VkFenceCreateInfo
import org.lwjgl.vulkan.VkInstance
import org.lwjgl.vulkan.VkInstanceCreateInfo
import org.lwjgl.vulkan.VkLayerProperties
import org.lwjgl.vulkan.VkMemoryAllocateInfo
import org.lwjgl.vulkan.VkMemoryRequirements
import org.lwjgl.vulkan.VkPhysicalDevice
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VkPhysicalDeviceProperties
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo
import org.lwjgl.vulkan.VkQueue
import org.lwjgl.vulkan.VkQueueFamilyProperties
import org.lwjgl.vulkan.VkShaderModuleCreateInfo
import org.lwjgl.vulkan.VkSubmitInfo
import org.lwjgl.vulkan.VkWriteDescriptorSet
import java.nio.ByteBuffer
import java.nio.IntBuffer
import kotlin.math.ceil

const val VK_LAYER_KHRONOS_VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"

val DEBUG = System.getProperty("debug", "false").toBoolean()

fun MemoryStack.pointers(strings: Iterable<String>): PointerBuffer =
    pointers(*strings.map { UTF8(it) }.toTypedArray())

const val PIXEL_SIZE = 4 * Float.SIZE_BYTES

class Vulkan(
    val shaderFile: String,
    val inputWidth: Int = 128,
    val inputHeight: Int = 128,
    val outputWidth: Int = 6400,
    val outputHeight: Int = 4800,
    val workgroupSize: Int = 32,
) {
    val inputBufferSize: Long = inputWidth * inputHeight * Float.SIZE_BYTES.toLong()
    val outputBufferSize: Long = outputWidth * outputHeight * PIXEL_SIZE.toLong()

    val enabledExtensions =
        if (DEBUG) listOf(
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        )
        else listOf()

    val enabledLayers =
        if (DEBUG) listOf(VK_LAYER_KHRONOS_VALIDATION_LAYER_NAME)
        else listOf()

    val supportedInstanceExtensions: List<String>
    val supportedInstanceLayers: List<String>
    val instance: VkInstance
    val physicalDevice: VkPhysicalDevice
    val queueFamilyIndex: Int
    val device: VkDevice
    val queue: VkQueue
    val inputBuffer: Long
    val inputBufferMemory: Long
    val outputBuffer: Long
    val outputBufferMemory: Long
    val descriptorSetLayout: Long
    val descriptorPool: Long
    val descriptorSet: Long
    val computeShaderModule: Long
    val pipelineLayout: Long
    val pipeline: Long
    val commandPool: Long
    val commandBuffer: VkCommandBuffer

    init {
        this.supportedInstanceExtensions = enumerateSupportedInstanceExtensions()
        this.supportedInstanceLayers = enumerateSupportedInstanceLayers()

        val unsupportedExtensions = enabledExtensions.filter { it !in supportedInstanceExtensions }
        val unsupportedLayers = enabledLayers.filter { it !in supportedInstanceLayers }
        if (unsupportedExtensions.isNotEmpty() || unsupportedLayers.isNotEmpty())
            throw IllegalStateException("Unsupported: extensions $unsupportedExtensions, layers $unsupportedLayers")

        this.instance = createInstance()
        this.physicalDevice = selectPhysicalDevice()

        val (queueFamilyIndex, device, queue) = createDeviceAndQueue()
        this.queueFamilyIndex = queueFamilyIndex
        this.device = device
        this.queue = queue

        val (inputBuffer, inputBufferMemory) = createInputBuffer()
        this.inputBuffer = inputBuffer
        this.inputBufferMemory = inputBufferMemory

        val (outputBuffer, outputBufferMemory) = createOutputBuffer()
        this.outputBuffer = outputBuffer
        this.outputBufferMemory = outputBufferMemory

        this.descriptorSetLayout = createDescriptorSetLayout()

        val (descriptorPool, descriptorSet) = createDescriptorSet()
        this.descriptorPool = descriptorPool
        this.descriptorSet = descriptorSet

        val (computeShaderModule, pipelineLayout, pipeline) = createComputePipeline()
        this.computeShaderModule = computeShaderModule
        this.pipelineLayout = pipelineLayout
        this.pipeline = pipeline

        val (commandPool, commandBuffer) = createCommandBuffer()
        this.commandPool = commandPool
        this.commandBuffer = commandBuffer
    }

    fun enumerateSupportedInstanceExtensions(): List<String> = stackPush().use { stack ->
        val pPropertyCount = stack.mallocInt(1)
        vkEnumerateInstanceExtensionProperties(null as ByteBuffer?, pPropertyCount, null)
            .orThrow("Failed to enumerate number of instance extensions")

        val propertyCount = pPropertyCount.get(0)
        if (propertyCount == 0) return listOf()

        val pProperties = VkExtensionProperties.malloc(propertyCount, stack)
        vkEnumerateInstanceExtensionProperties(null as ByteBuffer?, pPropertyCount, pProperties)
            .orThrow("Failed to enumerate instance extensions")

        val names = pProperties.map { it.extensionNameString() }

        if (DEBUG) {
            println("Supported instance extensions:")
            println(names.joinToString("\n") { "- $it" })
            println()
        }

        return names
    }

    fun enumerateSupportedInstanceLayers(): List<String> = stackPush().use { stack ->
        val pPropertyCount = stack.mallocInt(1)
        vkEnumerateInstanceLayerProperties(pPropertyCount, null)
            .orThrow("Failed to enumerate number of instance layers")

        val propertyCount = pPropertyCount.get(0)
        if (propertyCount == 0) return listOf()

        val pProperties = VkLayerProperties.malloc(propertyCount, stack)
        vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties)
            .orThrow("Failed to enumerate instance layers")

        if (DEBUG) {
            println("Supported instance layers:")
            println(pProperties.joinToString("\n") { "- " + it.layerNameString() + ": " + it.descriptionString() })
            println()
        }

        pProperties.map { it.layerNameString() }
    }

    fun createInstance(): VkInstance = stackPush().use { stack ->
        val appInfo = VkApplicationInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
            .pApplicationName(stack.UTF8("VulkanTest"))
            .applicationVersion(1)
            .apiVersion(VK_API_VERSION_1_0)

        val createInfo = VkInstanceCreateInfo.calloc(stack)
            .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
            .flags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
            .pApplicationInfo(appInfo)
            .ppEnabledExtensionNames(stack.pointers(enabledExtensions))
            .ppEnabledLayerNames(stack.pointers(enabledLayers))

        val pInstance = stack.mallocPointer(1)
        vkCreateInstance(createInfo, null, pInstance).orThrow("Failed to create instance")
        return VkInstance(pInstance.get(0), createInfo)
    }

    fun selectPhysicalDevice(): VkPhysicalDevice = stackPush().use { stack ->
        val deviceCountBuff = IntArray(1)
        vkEnumeratePhysicalDevices(instance, deviceCountBuff, null).orThrow()
        val deviceCount = deviceCountBuff[0]

        val pPhysicalDevices = PointerBuffer.allocateDirect(deviceCountBuff[0])
        vkEnumeratePhysicalDevices(instance, deviceCountBuff, pPhysicalDevices).orThrow()
        val physicalDevices = List(deviceCount) { i -> VkPhysicalDevice(pPhysicalDevices[i], instance) }

        if (DEBUG) {
            println("Found physical devices:")
            physicalDevices.forEach { device ->
                val props = VkPhysicalDeviceProperties.calloc()
                vkGetPhysicalDeviceProperties(device, props)

                val api = props.apiVersion()
                val vulkanVersion = "${VK_VERSION_MAJOR(api)}.${VK_VERSION_MINOR(api)}.${VK_VERSION_PATCH(api)}"

                println("- Device name:             ${props.deviceNameString()}")
                println("  Vulkan version:          $vulkanVersion")
                println("  Max shared compute mem:  ${props.limits().maxComputeSharedMemorySize() / 1024} KB")
                println()
            }
        }

        return physicalDevices[0]
    }

    fun createDeviceAndQueue(): Triple<Int, VkDevice, VkQueue> = stackPush().use { stack ->
        val queueFamilyIndex = getComputeQueueFamilyIndex()

        val queueCreateInfo = VkDeviceQueueCreateInfo.calloc(1) // TODO: Multiple queues?
            .`sType$Default`()
            .queueFamilyIndex(queueFamilyIndex)
            .pQueuePriorities(stack.floats(1f))

        val deviceCreateInfo = VkDeviceCreateInfo.calloc()
            .`sType$Default`()
            .pQueueCreateInfos(queueCreateInfo)
            .ppEnabledExtensionNames(stack.pointers(listOf(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)))

        val pDevice = stack.mallocPointer(1)
        vkCreateDevice(physicalDevice, deviceCreateInfo, null, pDevice).orThrow("Failed to create device")
        val device = VkDevice(pDevice.get(0), physicalDevice, deviceCreateInfo)

        val pQueue = stack.mallocPointer(1)
        vkGetDeviceQueue(device, queueFamilyIndex, 0, pQueue)
        val queue = VkQueue(pQueue.get(0), device)

        return Triple(queueFamilyIndex, device, queue)
    }

    fun getComputeQueueFamilyIndex(): Int = stackPush().use { stack ->
        val pQueueFamilyCount = stack.mallocInt(1)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyCount, null)
        val queueFamilyCount = pQueueFamilyCount.get(0)

        val queueFamilies = VkQueueFamilyProperties.calloc(queueFamilyCount)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyCount, queueFamilies)
        queueFamilies.forEachIndexed { i, queueFamily ->
            if (DEBUG)
                println("Found queue family: queue count = ${queueFamily.queueCount()}, flags = ${queueFamily.queueFlags()}")

            if (queueFamily.queueCount() > 0 && queueFamily.queueFlags() and VK_QUEUE_COMPUTE_BIT != 0)
                return i
        }

        throw IllegalStateException("No compute queue family was found")
    }

    fun createInputBuffer(): Pair<Long, Long> = stackPush().use { stack ->
        val createInfo = VkBufferCreateInfo.calloc()
            .`sType$Default`()
            .size(inputBufferSize)
            .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            .sharingMode(VK_SHARING_MODE_EXCLUSIVE)

        val pInputBuffer = stack.mallocLong(1)
        vkCreateBuffer(device, createInfo, null, pInputBuffer).orThrow()
        val inputBuffer = pInputBuffer.get(0)

        val memoryRequirements = VkMemoryRequirements.calloc()
        vkGetBufferMemoryRequirements(device, inputBuffer, memoryRequirements)

        val allocateInfo = VkMemoryAllocateInfo.calloc()
            .`sType$Default`()
            .allocationSize(memoryRequirements.size())
            .memoryTypeIndex(
                findMemoryType(
                    memoryRequirements.memoryTypeBits(),
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT or VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                )
            )

        val pInputBufferMemory = stack.mallocLong(1)
        vkAllocateMemory(device, allocateInfo, null, pInputBufferMemory).orThrow()
        val inputBufferMemory = pInputBufferMemory.get()

        vkBindBufferMemory(device, inputBuffer, inputBufferMemory, 0).orThrow()

        return Pair(inputBuffer, inputBufferMemory)
    }

    fun createOutputBuffer(): Pair<Long, Long> = stackPush().use { stack ->
        val bufferCreateInfo = VkBufferCreateInfo.calloc()
            .`sType$Default`()
            .size(outputBufferSize)
            .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            .sharingMode(VK_SHARING_MODE_EXCLUSIVE)

        val pBuffer = stack.mallocLong(1)
        vkCreateBuffer(device, bufferCreateInfo, null, pBuffer).orThrow()
        val buffer = pBuffer.get(0)

        val memoryRequirements = VkMemoryRequirements.calloc()
        vkGetBufferMemoryRequirements(device, buffer, memoryRequirements)

        val allocateInfo = VkMemoryAllocateInfo.calloc()
            .`sType$Default`()
            .allocationSize(memoryRequirements.size())
            .memoryTypeIndex(
                findMemoryType(
                    memoryRequirements.memoryTypeBits(),
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT or VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                )
            )

        val pBufferMemory = stack.mallocLong(1)
        vkAllocateMemory(device, allocateInfo, null, pBufferMemory).orThrow()
        val bufferMemory = pBufferMemory.get()

        vkBindBufferMemory(device, buffer, bufferMemory, 0).orThrow()

        return Pair(buffer, bufferMemory)
    }

    fun findMemoryType(memoryTypeBits: Int, properties: Int): Int = stackPush().use { stack ->
        val memoryProperties = VkPhysicalDeviceMemoryProperties.calloc()
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties)

        for (i in 0 until memoryProperties.memoryTypeCount())
            if (memoryTypeBits and (1 shl i) != 0 &&
                (memoryProperties.memoryTypes().get(i).propertyFlags() and properties) == properties
            ) return i

        return -1
    }

    fun createDescriptorSetLayout(): Long = stackPush().use { stack ->
        val bindings = VkDescriptorSetLayoutBinding.calloc(2)

        bindings.get(0)
            .binding(0)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)

        bindings.get(1)
            .binding(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)

        val createInfo = VkDescriptorSetLayoutCreateInfo.calloc()
            .`sType$Default`()
            .pBindings(bindings)

        val pDescriptorSetLayout = stack.mallocLong(1)
        vkCreateDescriptorSetLayout(device, createInfo, null, pDescriptorSetLayout).orThrow()
        return pDescriptorSetLayout.get(0)
    }

    fun createDescriptorSet(): Pair<Long, Long> = stackPush().use { stack ->
        val poolSize = VkDescriptorPoolSize.calloc(1)
            .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(2)

        val poolCreateInfo = VkDescriptorPoolCreateInfo.calloc()
            .`sType$Default`()
            .maxSets(1)
            .pPoolSizes(poolSize)

        val pDescriptorPool = stack.mallocLong(1)
        vkCreateDescriptorPool(device, poolCreateInfo, null, pDescriptorPool).orThrow()
        val descriptorPool = pDescriptorPool.get()

        val allocateInfo = VkDescriptorSetAllocateInfo.calloc()
            .`sType$Default`()
            .descriptorPool(descriptorPool)
            .pSetLayouts(stack.longs(descriptorSetLayout))

        val pDescriptorSets = stack.mallocLong(1)
        vkAllocateDescriptorSets(device, allocateInfo, pDescriptorSets).orThrow()
        val descriptorSet = pDescriptorSets.get(0)

        val writeDescriptorSet = VkWriteDescriptorSet.calloc(2)

        val inputBufferInfo = VkDescriptorBufferInfo.calloc(1)
            .buffer(inputBuffer)
            .offset(0)
            .range(inputBufferSize)

        writeDescriptorSet.get(0)
            .`sType$Default`()
            .dstSet(pDescriptorSets.get(0))
            .dstBinding(0)
            .descriptorCount(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .pBufferInfo(inputBufferInfo)

        val outputBufferInfo = VkDescriptorBufferInfo.calloc(1)
            .buffer(outputBuffer)
            .offset(0)
            .range(outputBufferSize)

        writeDescriptorSet.get(1)
            .`sType$Default`()
            .dstSet(pDescriptorSets.get(0))
            .dstBinding(1)
            .descriptorCount(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .pBufferInfo(outputBufferInfo)

        vkUpdateDescriptorSets(device, writeDescriptorSet, null)

        return Pair(descriptorPool, descriptorSet)
    }

    fun createComputePipeline(): Triple<Long, Long, Long> = stackPush().use { stack ->
        val code = readFileToNative(shaderFile)
        val createInfo = VkShaderModuleCreateInfo.calloc()
            .`sType$Default`()
            .pCode(code)

        val pComputeShaderModule = stack.mallocLong(1)
        vkCreateShaderModule(device, createInfo, null, pComputeShaderModule).orThrow()
        val computeShaderModule = pComputeShaderModule.get(0)
        memFree(code)

        val pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo.calloc()
            .`sType$Default`()
            .setLayoutCount(1)
            .pSetLayouts(stack.longs(descriptorSetLayout))

        val pPipelineLayout = stack.mallocLong(1)
        vkCreatePipelineLayout(device, pipelineLayoutCreateInfo, null, pPipelineLayout).orThrow()
        val pipelineLayout = pPipelineLayout.get(0)

        val shaderStageCreateInto = VkPipelineShaderStageCreateInfo.calloc()
            .`sType$Default`()
            .stage(VK_SHADER_STAGE_COMPUTE_BIT)
            .module(computeShaderModule)
            .pName(stack.UTF8("main"))

        val pipelineCreateInfo = VkComputePipelineCreateInfo.calloc(1)
            .`sType$Default`()
            .stage(shaderStageCreateInto)
            .layout(pipelineLayout)

        val pPipeline = stack.mallocLong(1)
        vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineCreateInfo, null, pPipeline).orThrow()
        val pipeline = pPipeline.get(0)

        return Triple(computeShaderModule, pipelineLayout, pipeline)
    }

    fun createCommandBuffer(): Pair<Long, VkCommandBuffer> = stackPush().use { stack ->
        val commandPoolCreateInfo = VkCommandPoolCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)
            .queueFamilyIndex(queueFamilyIndex)

        val pCommandPool = stack.mallocLong(1)
        vkCreateCommandPool(device, commandPoolCreateInfo, null, pCommandPool).orThrow()
        val commandPool = pCommandPool.get(0)

        val commandBufferAllocateInfo = VkCommandBufferAllocateInfo.calloc()
            .`sType$Default`()
            .commandPool(commandPool)
            .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
            .commandBufferCount(1)

        val pCommandBuffer = stack.mallocPointer(1)
        vkAllocateCommandBuffers(device, commandBufferAllocateInfo, pCommandBuffer).orThrow()
        val commandBuffer = VkCommandBuffer(pCommandBuffer.get(0), device)

        val beginInfo = VkCommandBufferBeginInfo.calloc()
            .`sType$Default`()
            .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)

        vkBeginCommandBuffer(commandBuffer, beginInfo).orThrow()

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            stack.longs(descriptorSet),
            null as IntBuffer?,
        )
        vkCmdDispatch(
            commandBuffer,
            ceil(outputWidth / workgroupSize.toFloat()).toInt(),
            ceil(outputHeight / workgroupSize.toFloat()).toInt(),
            1,
        )
        vkEndCommandBuffer(commandBuffer).orThrow()
        return Pair(commandPool, commandBuffer)
    }

    fun runCommandBuffer() = stackPush().use { stack ->
        val fenceCreateInfo = VkFenceCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)

        val pFence = stack.mallocLong(1)
        vkCreateFence(device, fenceCreateInfo, null, pFence).orThrow()
        val fence = pFence.get(0)

        val submitInfo = VkSubmitInfo.calloc()
            .`sType$Default`()
            .pCommandBuffers(stack.pointers(commandBuffer))

        vkQueueSubmit(queue, submitInfo, fence).orThrow()
        vkWaitForFences(device, fence, true, 100000000000).orThrow()
        vkDestroyFence(device, fence, null)
    }

    fun setInput(floatArray: FloatArray): Unit = stackPush().use { stack ->
        val inputSizeBytes = floatArray.size.toLong() * Float.SIZE_BYTES
        if (inputSizeBytes != inputBufferSize)
            throw IllegalArgumentException("Input must be $inputBufferSize bytes but got ${floatArray.size} * ${Float.SIZE_BYTES} = $inputSizeBytes")

        val pMappedMemory = stack.mallocPointer(1)
        vkMapMemory(device, inputBufferMemory, 0, inputBufferSize, 0, pMappedMemory).orThrow()
        val mappedMemory = pMappedMemory.get(0)
        val buffer = memByteBuffer(mappedMemory, inputBufferSize.toInt()).asFloatBuffer()

        buffer.put(floatArray)

        vkUnmapMemory(device, inputBufferMemory)
    }

    fun getRenderedImage(): FloatArray = stackPush().use { stack ->
        val pMappedMemory = stack.mallocPointer(1)
        vkMapMemory(device, outputBufferMemory, 0, outputBufferSize, 0, pMappedMemory).orThrow()
        val mappedMemory = pMappedMemory.get(0)
        val buffer = memByteBuffer(mappedMemory, outputBufferSize.toInt()).asFloatBuffer()

        val bytes = FloatArray(outputBufferSize.toInt() / Float.SIZE_BYTES)
        buffer.get(bytes)

        vkUnmapMemory(device, outputBufferMemory)
        return bytes
    }

    fun destroy() {
        vkFreeMemory(device, inputBufferMemory, null)
        vkFreeMemory(device, outputBufferMemory, null)
        vkDestroyBuffer(device, inputBuffer, null)
        vkDestroyBuffer(device, outputBuffer, null)
        vkDestroyShaderModule(device, computeShaderModule, null)
        vkDestroyDescriptorPool(device, descriptorPool, null)
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null)
        vkDestroyPipelineLayout(device, pipelineLayout, null)
        vkDestroyPipeline(device, pipeline, null)
        vkDestroyCommandPool(device, commandPool, null)
        vkDestroyDevice(device, null)
        vkDestroyInstance(instance, null)
    }
}
