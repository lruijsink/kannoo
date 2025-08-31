package kannoo.vulkan

import org.lwjgl.PointerBuffer
import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.vulkan.EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME
import org.lwjgl.vulkan.KHRGetPhysicalDeviceProperties2.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
import org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
import org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
import org.lwjgl.vulkan.KHRPortabilitySubset.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
import org.lwjgl.vulkan.VK10.VK_API_VERSION_1_0
import org.lwjgl.vulkan.VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
import org.lwjgl.vulkan.VK10.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
import org.lwjgl.vulkan.VK10.VK_QUEUE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.VK_SHARING_MODE_EXCLUSIVE
import org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_APPLICATION_INFO
import org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
import org.lwjgl.vulkan.VK10.VK_VERSION_MAJOR
import org.lwjgl.vulkan.VK10.VK_VERSION_MINOR
import org.lwjgl.vulkan.VK10.VK_VERSION_PATCH
import org.lwjgl.vulkan.VK10.vkAllocateMemory
import org.lwjgl.vulkan.VK10.vkBindBufferMemory
import org.lwjgl.vulkan.VK10.vkCreateBuffer
import org.lwjgl.vulkan.VK10.vkCreateDevice
import org.lwjgl.vulkan.VK10.vkCreateInstance
import org.lwjgl.vulkan.VK10.vkDestroyBuffer
import org.lwjgl.vulkan.VK10.vkDestroyDevice
import org.lwjgl.vulkan.VK10.vkDestroyInstance
import org.lwjgl.vulkan.VK10.vkEnumerateInstanceExtensionProperties
import org.lwjgl.vulkan.VK10.vkEnumerateInstanceLayerProperties
import org.lwjgl.vulkan.VK10.vkEnumeratePhysicalDevices
import org.lwjgl.vulkan.VK10.vkFreeMemory
import org.lwjgl.vulkan.VK10.vkGetBufferMemoryRequirements
import org.lwjgl.vulkan.VK10.vkGetDeviceQueue
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceProperties
import org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceQueueFamilyProperties
import org.lwjgl.vulkan.VkApplicationInfo
import org.lwjgl.vulkan.VkBufferCreateInfo
import org.lwjgl.vulkan.VkDevice
import org.lwjgl.vulkan.VkDeviceCreateInfo
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo
import org.lwjgl.vulkan.VkExtensionProperties
import org.lwjgl.vulkan.VkInstance
import org.lwjgl.vulkan.VkInstanceCreateInfo
import org.lwjgl.vulkan.VkLayerProperties
import org.lwjgl.vulkan.VkMemoryAllocateInfo
import org.lwjgl.vulkan.VkMemoryRequirements
import org.lwjgl.vulkan.VkPhysicalDevice
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties
import org.lwjgl.vulkan.VkPhysicalDeviceProperties
import org.lwjgl.vulkan.VkQueue
import org.lwjgl.vulkan.VkQueueFamilyProperties
import java.nio.ByteBuffer

const val VK_LAYER_KHRONOS_VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation"

val DEBUG = System.getProperty("debug", "false").toBoolean()

class Vulkan {
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
    }

    private fun enumerateSupportedInstanceExtensions(): List<String> = stackPush().use { stack ->
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

    private fun enumerateSupportedInstanceLayers(): List<String> = stackPush().use { stack ->
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

    private fun createInstance(): VkInstance = stackPush().use { stack ->
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

    private fun selectPhysicalDevice(): VkPhysicalDevice = stackPush().use { stack ->
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

    private fun createDeviceAndQueue(): Triple<Int, VkDevice, VkQueue> = stackPush().use { stack ->
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

    private fun getComputeQueueFamilyIndex(): Int = stackPush().use { stack ->
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

    fun createBuffer(size: Long): VulkanBuffer = stackPush().use { stack ->
        val createInfo = VkBufferCreateInfo.calloc()
            .`sType$Default`()
            .size(size)
            .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            .sharingMode(VK_SHARING_MODE_EXCLUSIVE)

        val pBuffer = stack.mallocLong(1)
        vkCreateBuffer(device, createInfo, null, pBuffer).orThrow()
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

        val pMemory = stack.mallocLong(1)
        vkAllocateMemory(device, allocateInfo, null, pMemory).orThrow()
        val memory = pMemory.get()

        vkBindBufferMemory(device, buffer, memory, 0).orThrow()

        return VulkanBuffer(handle = buffer, memory = memory)
    }

    private fun findMemoryType(memoryTypeBits: Int, properties: Int): Int = stackPush().use { stack ->
        val memoryProperties = VkPhysicalDeviceMemoryProperties.calloc()
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties)

        for (i in 0 until memoryProperties.memoryTypeCount())
            if (memoryTypeBits and (1 shl i) != 0 &&
                (memoryProperties.memoryTypes().get(i).propertyFlags() and properties) == properties
            ) return i

        return -1
    }

    fun destroyBuffer(buffer: VulkanBuffer) {
        vkFreeMemory(device, buffer.memory, null)
        vkDestroyBuffer(device, buffer.handle, null)
    }

    fun destroyBuffers(vararg buffers: VulkanBuffer) {
        buffers.forEach { destroyBuffer(it) }
    }

    fun destroy() {
        vkDestroyDevice(device, null)
        vkDestroyInstance(instance, null)
    }
}
