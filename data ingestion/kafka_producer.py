# filepath: backend/kafka_producer.py
from confluent_kafka import Producer

p = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

def produce_message(topic, message):
    p.produce(topic, message.encode('utf-8'), callback=delivery_report)
    p.flush()

# Example usage
produce_message('fake_news', 'This is a test message')