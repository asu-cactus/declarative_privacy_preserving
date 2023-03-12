
date_range = 365
location_range = 20

hidden_units = 1000

frequency_range = 1
training_size = 10000

learning_rate = 0.1 # Good for plan 2
# learning_rate = 0.01
epochs = 50

countries = ['UK', 'US', 'BRZ', 'CHN', 'AU']

assert training_size % frequency_range == 0
assert frequency_range < location_range
assert frequency_range < date_range
