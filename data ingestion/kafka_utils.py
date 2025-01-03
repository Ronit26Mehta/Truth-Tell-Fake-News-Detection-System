from confluent_kafka import Producer, Consumer, KafkaException
from config import Config

class KafkaProducer:
    def __init__(self):
        self.producer = Producer({'bootstrap.servers': Config.KAFKA_BOOTSTRAP_SERVERS})

    def delivery_report(self, err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

    def produce_message(self, topic, message):
        self.producer.produce(topic, message.encode('utf-8'), callback=self.delivery_report)
        self.producer.flush()

class KafkaConsumer:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': Config.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'mygroup',
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([Config.KAFKA_TOPIC])

    def consume_messages(self):
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(msg.error())
                        break
                yield msg.value().decode('utf-8')
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()