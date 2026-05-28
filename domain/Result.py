from typing import Any
from pydantic import BaseModel

class Result(BaseModel):
    # 字段必须在类级别定义
    code: int = None
    msg: str = None
    data: Any = None  # 任意类型的数据

    def __init__(self, **kwargs):
        # 调用父类的 __init__ 方法，确保 Pydantic 的字段验证机制正常工作
        super().__init__(**kwargs)
        # 这里可以添加自定义的初始化逻辑
        # 注意：不需要再给字段赋值，Pydantic 会自动处理

    def ok(self, data: Any):
        self.code = 200
        self.msg = "OK"
        self.data = data
        return self

    def error(self, msg: str):
        self.code = 401
        self.msg = msg  # 错误信息
        return self

R = Result()