import threading

def thread_func(id):
    f.write(f'{id}\n')

with open('test_file.txt', 'w') as f:
    thread1 = threading.Thread(target=thread_func, args=('thread1',))
    thread2 = threading.Thread(target=thread_func, args=('thread2',))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()