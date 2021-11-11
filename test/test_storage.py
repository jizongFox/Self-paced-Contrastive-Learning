import random

from contrastyou.storage import Storage

storage = Storage()
for epoch in range(0,10,2):
    report_dict = {"loss":1, "dice_meter":{"dice": random.random(), "dice2": random.random()}}
    if epoch >5:
        report_dict.update({"dice_6":random.random()})
    storage.put_all(report_dict, epoch=epoch)
print(storage["loss"].summary())
print(storage["dice_meter"].summary())
print(storage["dice_6"].summary())
print(storage.summary())


