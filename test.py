import queue
import threading

# Create a message queue
message_queue = queue.Queue()

# Define a function that will put messages into the queue
def producer():
    while True:
        message = input("Enter a message: ")
        message_queue.put(message)


# Define a function that will consume messages from the queue
def consumer():
    while True:
        message = message_queue.get()
        print(f"Got message: {message}")


# Create two threads, one for the producer and one for the consumer
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

# Start the threads
producer_thread.start()
consumer_thread.start()
