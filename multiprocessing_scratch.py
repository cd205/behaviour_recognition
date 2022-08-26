# import time
# from multiprocessing import Process, Value, Lock

# def func(val, lock):
#     for i in range(50):
#         time.sleep(0.01)
#         with lock:
#             val.value += 1

# if __name__ == '__main__':
#     v = Value('i', 0)
#     lock = Lock()
#     procs = [Process(target=func, args=(v, lock)) for i in range(10)]

#     for p in procs: p.start()
#     for p in procs: p.join()

#     print(v.value)
################################
# import multiprocessing
  
# def print_records(records):
#     """
#     function to print record(tuples) in records(list)
#     """
#     for record in records:
#         print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))
  
# def insert_record(record, records):
#     """
#     function to add a new record to records(list)
#     """
#     records.append(record)
#     print("New record added!\n")
  
# if __name__ == '__main__':
#     with multiprocessing.Manager() as manager:
#         # creating a list in server process memory
#         records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)])
#         # new record to be inserted in records
#         new_record = ('Jeff', 8)
  
#         # creating new processes
#         p1 = multiprocessing.Process(target=insert_record, args=(new_record, records))
#         p2 = multiprocessing.Process(target=print_records, args=(records,))
  
#         # running process p1 to insert new record
#         p1.start()
#         p1.join()
  
#         # running process p2 to print records
#         p2.start()
#         p2.join()



# #######################
# import multiprocessing
# import time
# import numpy as np

# def print_records(records):
#     """
#     function to print record(tuples) in records(list)
#     """

#     n = 0
#     while n<5:
#         print(records)
#         n+=1
#         time.sleep(1)
#     #for record in records:
#     #    print(record[1])
#         #print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))
  
# def insert_record(record, records):
#     """
#     function to add a new record to records(list)
#     """
    
#     #plus1 = record + 1
#     #records.append(plus1)
#     n = 0
#     while n<5:
#         val = np.random.randint(10)
#         #records.append(val)
#         records = val
#         print("New record added: ", val)
#         n+=1
#         time.sleep(1)
    

    
  
# if __name__ == '__main__':
#     with multiprocessing.Manager() as manager:
#         # creating a list in server process memory
#         #records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)])
#         records = manager.Value('i',100)
#         # new record to be inserted in records
#         new_record = (8)
  
#         # creating new processes
#         p1 = multiprocessing.Process(target=insert_record, args=(new_record, records))
#         p2 = multiprocessing.Process(target=print_records, args=(records,))
  
      
#         p1.start()
#         p2.start()

#         p1.join()
#         p2.join()




####################### USING LOCK
import multiprocessing
import time
import numpy as np

def print_records(records):
    """
    function to print values
    """
    n = 0
    while n<5:
        print(records.value)
        n+=1
        time.sleep(1)

  
def insert_record(records, lock):
    """
    function to change value
    """
    n = 0
    while n<5:
        with lock:
            val = np.random.randint(10)
            records.value = val
        print("Value changed to: ", val)
        n+=1
        time.sleep(1)
    

    
  
if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # creating a value in server process memory
        lock = manager.Lock()
        records = manager.Value('int',1.0)
  
        # creating new processes
        p1 = multiprocessing.Process(target=insert_record, args=(records, lock))
        p2 = multiprocessing.Process(target=print_records, args=(records,))
      
        p1.start()
        p2.start()
        p1.join()
        p2.join()