package kannoo.vulkan

class Shader(fileName: String, val workgroupSize: Int) {
    val code = readFileToNative(fileName)
}
