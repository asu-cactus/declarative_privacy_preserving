
date_range = 365
location_range = 20

hidden_units = 500

frequency_range = 10
training_size = 1000

learning_rate = 0.1
epochs = 100

countries = ['UK', 'US', 'BRZ', 'CHN', 'AU']

assert training_size % frequency_range == 0
assert frequency_range < location_range
assert frequency_range < date_range