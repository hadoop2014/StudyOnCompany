import random
import time
import multiprocessing
import os

def printmesg(i,ans):
    print(os.getpid(), ':', i, 'sucess result is', ans)


def lock_raise_error(i,l):
    try:
        rand=random.randrange(0,2)+1
        #with l:
        #l.acquire()
        with l:
            ans=10/rand
        #time.sleep(1)
        #print(os.getpid(),':',i,'sucess result is',ans)
            printmesg(i,ans)
        #l.release()
        return ans
    except Exception as e:
        print(os.getpid(),'failed',e)
        return -1

def testMultiprocessing():
    print('Parent process %s.' % os.getpid())
    pool = multiprocessing.Pool(processes=3)
    lock = multiprocessing.Manager().Lock()  # 使用Manager加锁
    result = []
    for i in range(15):
        a = pool.apply_async(func=lock_raise_error, args=(i, lock))
        result.append(a)
    pool.close()
    pool.join()
    print(a.successful())
    print([a.get() for a in result])

    processList = []
    for i in range(15):
        process = multiprocessing.Process(target=lock_raise_error,args = (i,lock))
        process.start()
        processList.append(process)

    for proces in processList:
        proces.join()

    print('done')


if __name__ == '__main__':
    testMultiprocessing()