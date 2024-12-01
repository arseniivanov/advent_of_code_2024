#define VULKAN_HPP_NO_CONSTRUCTORS
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <vulkan/vulkan.hpp>

struct QueueFamilyIndices {
  std::optional<uint32_t> computeFamily;
  bool isComplete() const { return computeFamily.has_value(); }
};

class MinimalCompute {
private:
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::Queue computeQueue;
  uint32_t computeQueueFamily;

  vk::CommandPool commandPool;
  vk::CommandBuffer commandBuffer;

  vk::DescriptorSetLayout descriptorSetLayout;
  vk::DescriptorPool descriptorPool;
  vk::DescriptorSet descriptorSet;

  vk::Pipeline computePipeline;
  vk::PipelineLayout pipelineLayout;

  // Storage buffers
  vk::Buffer inputBuffer;
  vk::Buffer outputBuffer;
  vk::DeviceMemory inputMemory;
  vk::DeviceMemory outputMemory;

  void createInstance() {
    // First create the application info
    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "Compute Shader Barebones";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Then create the instance info
    vk::InstanceCreateInfo createInfo;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;

    // Create the instance
    instance = vk::createInstance(createInfo);
  }

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    device.getQueueFamilyProperties(&queueFamilyCount, nullptr);
    std::vector<vk::QueueFamilyProperties> queueFamilies(queueFamilyCount);
    device.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
        indices.computeFamily = i;
        break;
      }
      i++;
    }

    return indices;
  }

  bool isDeviceSuitable(vk::PhysicalDevice device) {
    vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool isDiscrete =
        deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;

    return indices.isComplete() && isDiscrete;
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }
    if (!physicalDevice) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void createDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.queueFamilyIndex = indices.computeFamily.value();
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    vk::PhysicalDeviceFeatures deviceFeatures;

    vk::DeviceCreateInfo createInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = nullptr;

    device = physicalDevice.createDevice(createInfo);

    computeQueue = device.getQueue(indices.computeFamily.value(), 0);
  }

  void createDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings;

    // Input and output buffer spec
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
    layoutBindings[0].pImmutableSamplers = nullptr;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
    layoutBindings[1].pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
  }

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    return device.createShaderModule(createInfo);
  }

  void createComputePipeline() {
    auto computeShaderCode = readFile("shaders/comp.spv");
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
    computeShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    vk::ComputePipelineCreateInfo pipelineInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    computePipeline =
        device.createComputePipelines(nullptr, {pipelineInfo}).value[0];

    device.destroyShaderModule(computeShaderModule, nullptr);
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

    commandPool = device.createCommandPool(poolInfo);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties =
        physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                    vk::DeviceMemory &bufferMemory) {
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    buffer = device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements =
        device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    bufferMemory = device.allocateMemory(allocInfo);

    device.bindBufferMemory(buffer, bufferMemory, 0);
  }

  void createBuffer() {}

  void init() {
    createInstance();
    pickPhysicalDevice();
    createDevice();
    createDescriptorSetLayout();
    createComputePipeline();
    createCommandPool();
    createBuffer();
    // allocateBufferMemory();
    // createDescriptorPool();
    // createDesctiptorSets();
    // createCommandBuffer();
    // recordCommandBuffer();
  };

  void compute() {};
  void cleanup() {};

public:
  void run() {
    init();
    compute();
    cleanup();
  }
};

int main() {
  MinimalCompute app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
