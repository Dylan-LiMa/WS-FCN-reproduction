import time

class Timer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置计时器"""
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def step(self):
        """记录一个时间步"""
        current_time = time.time()
        step_time = current_time - self.last_time
        self.last_time = current_time
        return step_time
    
    def total_time(self):
        """获取总时间"""
        return time.time() - self.start_time
    
    def average_time(self, steps):
        """计算平均时间"""
        return self.total_time() / steps if steps > 0 else 0 