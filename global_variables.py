
frequency_range = 10
date_range = 365
location_range = 20

hidden_units = 500

training_size = 10000

learning_rate = 0.02
epochs = 200

countries = ['UK', 'US', 'BRZ', 'CHN', 'AU']

assert training_size % frequency_range == 0
assert frequency_range < location_range
assert frequency_range < date_range