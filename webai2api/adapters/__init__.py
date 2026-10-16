"""适配器包。

每个适配器负责把"一家 AI 网页版"封装成统一的内部接口，供网关调用。
当前已实现：deepseek。
后续可在此新增 glm / kimi / qwen 等。

来源声明：deepseek 适配器基于开源项目 deepseek2api / deepseek2api-fixed
（MIT License）的内核代码，经集成改造而来。
"""
