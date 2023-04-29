import asyncio
import requests
import time

def fetch(url):
    time.sleep(1)
    print(f'printed {url}')
    return f'returned {url}'

async def main():
    urls = [
        'https://www.example.com/page1',
        'https://www.example.com/page2',
        'https://www.example.com/page3',
        'https://www.example.com/page4'
    ]
    loop = asyncio.get_event_loop()

    futures = []
    for url in urls:
        futures.append(loop.run_in_executor(None, fetch, url))
    
    for f in futures:
        ret = await f
        print(ret)
        
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())