from batch_data import BatchData

input_path = '/home/simon/mathisi2/attention_proteins/data'
batch_size = 1
n_workers = 1
validation_split = 0.2
shuffle_dataset = True

batch = BatchData(input_path, batch_size, n_workers, validation_split, shuffle_dataset)
batch.dataset()
test,validate = batch.batch_data()

# print('=' * 83)
# dataiter = iter(x)
# data = dataiter.next()
# features, labels = data
# print(f"features: \n{features}\n"
#       f"labels: \n{labels}\n"
#       f"len features: {len(features)}\n"
#       f"len labels: {len(labels)}\n")
# print('=' * 83)

