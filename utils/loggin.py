import psutil

def log_memory_usage():
    process = psutil.Process()
    print(f'{process.memory_info().rss / 1024 / 1024/ 1024}gb')  
